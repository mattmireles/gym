"""
ExoGym Trainer - High-Level Distributed Training Orchestrator

This module provides the main training orchestration layer for ExoGym's distributed
machine learning framework. It handles multiprocessing, model state management,
and device-agnostic training setup.

## Key Components

### TrainingConfig
Configuration dataclass that holds all training parameters for serialization
across process boundaries. This is crucial for the multiprocessing spawn pattern
used by Trainer.fit().

### Trainer
Abstract base class that defines the distributed training interface. Subclasses
implement _build_connection() to set up distributed communication backends.

### LocalTrainer  
Concrete implementation for single-machine distributed training. Automatically
detects and configures CUDA, MPS, or CPU backends with appropriate distributed
communication libraries (NCCL for CUDA, Gloo for MPS/CPU).

## Multiprocessing Architecture

The training uses mp.spawn() for notebook safety and cross-platform compatibility:

1. **Main Process**: Creates TrainingConfig, spawns worker processes, collects results
2. **Worker Processes**: Each runs _worker() function with TrainNode training loop
3. **Result Collection**: Model states are averaged across all workers after training

## Data Flow

```
Trainer.fit() → TrainingConfig → mp.spawn() → _worker() × N processes
     ↓                                            ↓
Model Averaging ← result_queue ← TrainNode.train() × N
     ↓
Final Model
```

## Called by:
- Example scripts (example/nanogpt.py, example/mnist.py)  
- User training scripts importing LocalTrainer

## Calls:
- TrainNode for individual process training execution
- Strategy classes for distributed communication patterns
- torch.multiprocessing for process management

## Hardware Compatibility

- **CUDA**: Uses NCCL backend, automatic GPU device assignment
- **MPS**: Uses Gloo backend with CPU fallback for unsupported operations  
- **CPU**: Uses Gloo backend for CPU-only distributed training

## Critical Implementation Details

- Deep copy models to CPU before pickling for mp.spawn() to avoid GPU memory sharing
- Use OrderedDict for deterministic model state averaging
- Increment port numbers to avoid conflicts in repeated training runs
- Handle both direct datasets and dataset factory functions for data parallelism
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exogym.train_node import TrainNode
from exogym.strategy import Strategy

import os
from abc import abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Callable
from collections import OrderedDict


class TrainingConstants:
    """
    Named constants for distributed training configuration.
    
    These constants replace magic numbers throughout the training system with
    well-documented, meaningful values. Each constant includes explanation of
    its purpose and reasoning behind the chosen value.
    """
    
    # Network Communication
    DEFAULT_MASTER_PORT = 12355
    """
    Default starting port for distributed training communication.
    
    PyTorch distributed requires a master port for process coordination. This port
    is automatically incremented for subsequent training runs to avoid conflicts.
    
    Port 12355 is chosen because:
    - It's in the dynamic/private port range (49152-65535 on most systems)
    - It's unlikely to conflict with system services
    - It's memorable and easy to debug in network tools
    
    Note: Actual port used is DEFAULT_MASTER_PORT + run_number to avoid conflicts
    """
    
    # Default Training Configuration
    DEFAULT_BATCH_SIZE = 16
    """
    Default effective batch size across minibatch accumulation.
    
    This represents the total batch size after accumulating gradients across
    multiple forward passes. Chosen as a reasonable starting point that:
    - Fits in memory for most models on modern GPUs
    - Provides stable gradient estimates for optimization
    - Can be scaled up based on available memory
    """
    
    DEFAULT_MINIBATCH_SIZE = 16
    """
    Default per-forward-pass batch size for memory management.
    
    This is the actual batch size used in each forward/backward pass, with
    gradients accumulated to reach the effective batch size. Chosen to:
    - Fit comfortably in 8GB+ GPU memory for typical models
    - Allow gradient accumulation for larger effective batch sizes
    - Provide reasonable training speed without excessive memory pressure
    """
    
    DEFAULT_VAL_SIZE = 64
    """
    Default number of samples for validation evaluation.
    
    This determines how many validation samples are used for evaluation during
    training. The value balances:
    - Statistical significance of validation metrics
    - Evaluation time overhead during training
    - Memory usage during validation passes
    
    64 samples typically provides stable validation metrics while maintaining
    fast evaluation cycles.
    """
    
    DEFAULT_VAL_INTERVAL = 100
    """
    Default number of training steps between validation evaluations.
    
    Validation is performed every VAL_INTERVAL training steps. This interval
    balances:
    - Training speed (less frequent evaluation = faster training)
    - Monitoring granularity (more frequent = better loss curves)
    - Compute overhead (validation requires forward passes on validation data)
    
    100 steps provides good monitoring without significant slowdown.
    """
    
    DEFAULT_CHECKPOINT_INTERVAL = 100
    """
    Default number of training steps between checkpoint saves.
    
    Checkpoints are saved every CHECKPOINT_INTERVAL steps to enable training
    resumption. This interval balances:
    - Recovery time in case of failures (more frequent = less work lost)
    - I/O overhead (checkpointing writes large files to disk)
    - Storage space (more frequent = more disk usage)
    
    100 steps is frequent enough to avoid losing significant work while
    maintaining reasonable I/O overhead.
    """

def print_dataset_size(dataset: torch.utils.data.Dataset):
    import pickle
    import io

    buffer = io.BytesIO()
    pickle.dump(dataset, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dataset size: {buffer.tell() // 1024 // 1024} MB")


@dataclass
class TrainingConfig:
    """
    Serializable configuration container for distributed training parameters.
    
    This dataclass packages all training parameters for safe serialization across
    process boundaries during multiprocessing spawn. All fields must be pickleable
    since they will be transferred to worker processes.
    
    ## Critical Design Requirements
    
    ### Serialization Safety
    - Model must be CPU-resident to avoid GPU memory sharing conflicts
    - All fields must be pickleable (no CUDA contexts, file handles, etc.)
    - Complex objects should be deep-copied before inclusion
    
    ### Dataset Flexibility
    - Supports direct datasets for simple data parallelism
    - Supports dataset factory functions for advanced partitioning
    - Factory pattern: callable(rank, num_nodes, is_val) -> Dataset
    
    ## Parameter Categories
    
    ### Core Training Configuration
    - model: CPU copy of the model to be trained
    - strategy: Communication strategy (copied independently per process)
    - num_epochs/max_steps: Training duration specification
    
    ### Data Pipeline Configuration  
    - train_dataset/val_dataset: Data sources (direct or factory)
    - batch_size: Effective batch size across minibatch accumulation
    - minibatch_size: Per-forward-pass batch size (memory constraint)
    - shuffle: Whether to shuffle training data (handled by sampler or dataset)
    
    ### Hardware Configuration
    - device: Target device type ("cuda", "mps", "cpu", or None for auto-detect)
    - devices: Specific device IDs for multi-GPU setups
    - num_nodes: Number of distributed training processes
    
    ### Monitoring Configuration
    - val_size: Number of samples for validation evaluation
    - val_interval: Steps between validation runs
    - checkpoint_interval: Steps between checkpoint saves
    
    ### Training Optimizations
    - autocast: Enable automatic mixed precision training
    - trainer_class: Class type for trainer instantiation in worker processes
    - kwargs: Additional trainer-specific arguments
    
    ## Usage in Multiprocessing
    
    This configuration is created in the main process and passed to mp.spawn():
    1. Main process creates TrainingConfig with CPU model
    2. mp.spawn() serializes config and sends to worker processes  
    3. Worker processes deserialize and create trainer instances
    4. _worker() function reconstructs training environment from config
    
    ## Memory Considerations
    
    - Model should be deepcopied to CPU before config creation
    - Dataset factories are preferred over large dataset objects
    - Strategy objects are copied per process to maintain independence
    """

    model: torch.nn.Module
    train_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    val_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    strategy: Strategy
    num_epochs: int
    num_nodes: int
    max_steps: Optional[int] = None
    device: Optional[str] = None
    devices: Optional[List[int]] = None
    batch_size: int = TrainingConstants.DEFAULT_BATCH_SIZE
    minibatch_size: int = TrainingConstants.DEFAULT_MINIBATCH_SIZE
    shuffle: bool = True
    val_size: int = TrainingConstants.DEFAULT_VAL_SIZE
    val_interval: int = TrainingConstants.DEFAULT_VAL_INTERVAL
    autocast: bool = False
    checkpoint_interval: int = TrainingConstants.DEFAULT_CHECKPOINT_INTERVAL
    trainer_class: type = None
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize kwargs dict if None to prevent serialization issues."""
        if self.kwargs is None:
            self.kwargs = {}


def _worker(rank: int, config: TrainingConfig, result_queue: mp.Queue):
    """
    Distributed training worker process entry point.
    
    This function executes in each spawned child process and represents the core
    distributed training workflow. It must be importable at module level to be
    compatible with multiprocessing spawn method used by Jupyter notebooks.
    
    ## Process Lifecycle
    
    ### 1. Worker Initialization  
    - Deserialize TrainingConfig from parent process
    - Create trainer instance using config.trainer_class
    - Transfer all configuration parameters to trainer instance
    
    ### 2. Training Execution
    - Execute trainer._fit_process(rank) which handles:
      * Distributed communication setup
      * TrainNode creation and training loop
      * Strategy coordination with other processes
    
    ### 3. Result Collection
    - Extract final model state from completed training
    - Convert all tensors to CPU to avoid GPU serialization issues
    - Place (rank, state_dict) tuple in result queue for parent process
    
    ## Critical Implementation Details
    
    ### Memory Safety
    - All model tensors are detached and moved to CPU before serialization
    - OrderedDict ensures deterministic parameter ordering for averaging
    - CUDA contexts are explicitly avoided during serialization
    
    ### Rank Assignment
    - Each worker process receives unique rank ID (0 to num_nodes-1)
    - Rank determines distributed communication role and responsibilities
    - Rank 0 typically handles logging, rank 1 handles global evaluation
    
    ### Configuration Transfer
    - TrainingConfig is fully deserialized in worker process
    - All config fields are transferred to trainer instance attributes
    - Strategy and model objects are deep-copied to avoid shared state
    
    ### Error Handling
    - Worker process failures are isolated and don't crash main process
    - Failed workers will not put results in queue, causing main process timeout
    - CUDA/device errors are contained within worker processes
    
    ## Multiprocessing Compatibility
    
    ### Notebook Safety
    - Function is defined at module level for pickle compatibility
    - No closure variables or complex state dependencies
    - Compatible with both spawn and fork multiprocessing methods
    
    ### Spawn Method Benefits
    - Clean process separation without shared memory conflicts
    - Avoids CUDA context sharing issues between processes
    - Better compatibility across platforms (Windows, macOS, Linux)
    
    ## Communication Flow
    
    ```
    Parent Process → mp.spawn() → _worker() × N processes
         ↓                           ↓
    TrainingConfig → Deserialization → Trainer Instance
         ↓                           ↓  
    Result Queue ← CPU State Dict ← Training Complete
    ```
    
    Args:
        rank: Process rank ID (0 to num_nodes-1) for distributed coordination
        config: Complete training configuration with all parameters
        result_queue: Multiprocessing queue for returning final model state
        
    Returns:
        None (results placed in result_queue)
        
    Called by:
        torch.multiprocessing.spawn() from Trainer.fit()
        
    Calls:
        trainer._fit_process() for actual training execution
    """
    # Create trainer instance in the worker process
    trainer = config.trainer_class(
        model=config.model,
        train_dataset=config.train_dataset,
        val_dataset=config.val_dataset,
        **config.kwargs,
    )

    # Set all the configuration parameters
    trainer.num_epochs = config.num_epochs
    trainer.max_steps = config.max_steps
    trainer.strategy = config.strategy
    trainer.num_nodes = config.num_nodes
    trainer.device = config.device
    trainer.devices = config.devices
    trainer.batch_size = config.batch_size
    trainer.minibatch_size = config.minibatch_size
    trainer.shuffle = config.shuffle
    trainer.val_size = config.val_size
    trainer.val_interval = config.val_interval
    trainer.autocast = config.autocast
    trainer.checkpoint_interval = config.checkpoint_interval

    # Run the training process and get the final model state dict
    final_model_state = trainer._fit_process(rank)

    # Move tensors to CPU and detach to avoid CUDA serialization issues
    cpu_state_dict = OrderedDict()
    for key, tensor in final_model_state.items():
        cpu_state_dict[key] = tensor.detach().cpu()

    # Put the result in the queue
    result_queue.put((rank, cpu_state_dict))

def _average_model_states(model_states: Dict[int, OrderedDict]) -> OrderedDict:
    """
    Compute element-wise average of model state dictionaries from multiple processes.
    
    This function implements the final aggregation step in distributed training,
    combining the individually trained models from each worker process into a
    single averaged model. This averaging is crucial for distributed training
    convergence and represents the "model parallel" aspect of the training.
    
    ## Mathematical Operation
    
    For each parameter tensor p in the model:
    ```
    p_averaged = (1/N) * Σ(p_i) for i in [0, N-1]
    ```
    
    Where N is the number of worker processes and p_i is the parameter from
    worker process i after training completion.
    
    ## Implementation Details
    
    ### Tensor Stacking and Averaging
    - Uses torch.stack() to create a new dimension for process-wise parameters
    - torch.mean(dim=0) performs element-wise averaging across the process dimension
    - Preserves original tensor shapes and dtypes
    
    ### Parameter Ordering
    - Relies on OrderedDict to ensure consistent parameter ordering across processes
    - All model_states must have identical parameter names and shapes
    - Uses first state dict as template for parameter iteration
    
    ### Memory Efficiency
    - Creates temporary stacked tensors only during averaging operation
    - Final averaged state contains same memory footprint as single model
    - No persistent storage of individual process states
    
    ## Error Conditions
    
    ### Empty Input Handling
    - Returns None if no model states provided (should not happen in normal operation)
    - Gracefully handles edge case of process failures
    
    ### Shape Mismatches
    - torch.stack() will raise RuntimeError if parameter shapes differ between processes
    - This indicates a bug in distributed training synchronization
    - Such errors should be treated as fatal training failures
    
    ## Usage in Training Pipeline
    
    ```
    # Collect states from all worker processes
    model_states = {rank: state_dict for rank, state_dict in result_queue}
    
    # Average across all processes  
    averaged_state = _average_model_states(model_states)
    
    # Load into final model
    final_model.load_state_dict(averaged_state)
    ```
    
    Args:
        model_states: Dictionary mapping rank → model state dict from each worker
        
    Returns:
        OrderedDict containing averaged parameters, or None if no states provided
        
    Called by:
        Trainer.fit() after collecting results from all worker processes
        
    Raises:
        RuntimeError: If parameter shapes differ between processes (indicates bug)
    """
    if not model_states:
        return None

    # Get the first state dict as template
    averaged_state = OrderedDict()
    first_state = list(model_states.values())[0]

    # Average each parameter
    for param_name in first_state.keys():
        # Stack all versions of this parameter
        param_stack = torch.stack(
            [state[param_name] for state in model_states.values()]
        )
        # Average them
        averaged_state[param_name] = torch.mean(param_stack, dim=0)

    return averaged_state


class Trainer:
    """
    Abstract base class for distributed machine learning training.
    
    The Trainer class provides the high-level interface for distributed training
    orchestration, handling multiprocessing, model state management, and device
    configuration. Subclasses implement specific distributed communication setup
    through the _build_connection() method.
    
    ## Architecture Overview
    
    ### Multiprocessing Design
    - Uses torch.multiprocessing.spawn() for notebook safety and platform compatibility
    - Spawns N worker processes where each executes the complete training loop
    - Parent process coordinates workers and aggregates final model states
    
    ### Model State Management
    - Original model preserved on CPU in parent process
    - Each worker receives deep copy of model for independent training
    - Final models are averaged element-wise across all workers
    
    ### Device Abstraction
    - Supports CUDA, MPS (Apple Silicon), and CPU training
    - Automatic device detection and configuration in worker processes
    - Distributed communication backend selection (NCCL vs Gloo)
    
    ## Key Responsibilities
    
    ### Dataset Handling
    - Supports both direct datasets and dataset factory functions
    - Factory pattern enables advanced data partitioning strategies
    - Automatic distributed sampling for standard datasets
    
    ### Port Management
    - Automatically increments port numbers to avoid conflicts
    - Essential for running multiple training jobs on same machine
    - Configurable starting port for custom network environments
    
    ### Configuration Serialization
    - Packages all training parameters into TrainingConfig for worker processes
    - Ensures all parameters are properly transferred across process boundaries
    - Handles complex object serialization requirements
    
    ## Subclass Implementation
    
    Subclasses must implement:
    - `_build_connection()`: Set up distributed communication backend
    
    Common implementations:
    - LocalTrainer: Single-machine distributed training
    - RemoteTrainer: Multi-machine cluster training (future)
    
    ## Usage Pattern
    
    ```python
    trainer = LocalTrainer(model, train_dataset, val_dataset)
    final_model = trainer.fit(
        num_epochs=10,
        strategy=DiLoCoStrategy(H=100),
        num_nodes=4,
        batch_size=32
    )
    ```
    
    ## State Management
    
    ### Port Counter
    - Incremented with each fit() call to avoid port conflicts
    - Allows multiple sequential training runs in same process
    - Reset only when trainer instance is recreated
    
    ### Original Model Preservation
    - model_orig maintains original model state
    - Never modified during training - only used as template
    - Enables multiple training runs with same initial state
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        val_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        start_port: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize trainer with model, datasets, and communication configuration.
        
        Args:
            model: PyTorch model to train (will be deep-copied for training)
            train_dataset: Training data source (direct dataset or factory function)
            val_dataset: Validation data source (direct dataset or factory function)
            start_port: Starting port for distributed communication (auto-incremented)
            **kwargs: Additional trainer-specific configuration
        """
        self.model_orig = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.kwargs = kwargs
        self.port = start_port if start_port is not None else TrainingConstants.DEFAULT_MASTER_PORT

    def fit(
        self,
        num_epochs: int,
        strategy: Strategy,
        num_nodes: int,
        max_steps: int = None,
        device: str = None,
        devices: list[int] = None,
        batch_size: int = TrainingConstants.DEFAULT_BATCH_SIZE,
        minibatch_size: int = TrainingConstants.DEFAULT_MINIBATCH_SIZE,
        shuffle: bool = True,
        val_size: int = TrainingConstants.DEFAULT_VAL_SIZE,
        val_interval: int = TrainingConstants.DEFAULT_VAL_INTERVAL,
        autocast: bool = False,
        checkpoint_interval: int = TrainingConstants.DEFAULT_CHECKPOINT_INTERVAL,
        **kwargs,
    ):
        """
        Execute distributed training with specified configuration and strategy.
        
        This is the main entry point for distributed training. It handles the complete
        training lifecycle from process spawning to final model aggregation, returning
        a trained model that represents the averaged result from all worker processes.
        
        ## Training Process Overview
        
        ### 1. Configuration and Validation
        - Validates training parameters (val_size ≥ batch_size, etc.)
        - Creates TrainingConfig with all parameters for worker processes
        - Increments port number to avoid conflicts with previous runs
        
        ### 2. Model Preparation
        - Deep copies original model to CPU for safe multiprocessing serialization
        - Avoids GPU memory sharing issues between parent and worker processes
        - Preserves original model state for potential future training runs
        
        ### 3. Worker Process Spawning
        - Uses torch.multiprocessing.spawn() for notebook compatibility
        - Spawns num_nodes worker processes, each with unique rank
        - Each worker executes _worker() function with complete training loop
        
        ### 4. Result Collection and Aggregation
        - Collects final model states from all worker processes via result queue
        - Performs element-wise averaging of model parameters across workers
        - Loads averaged state into a copy of the original model
        
        ## Parameter Categories
        
        ### Training Duration
        - num_epochs: Number of complete dataset passes
        - max_steps: Optional limit on total training steps (overrides epochs)
        
        ### Distributed Configuration
        - strategy: Communication strategy (e.g., DiLoCoStrategy, SimpleReduceStrategy)
        - num_nodes: Number of distributed worker processes
        - device/devices: Hardware configuration and GPU assignment
        
        ### Data Pipeline
        - batch_size: Effective batch size after gradient accumulation
        - minibatch_size: Per-forward-pass batch size for memory management
        - shuffle: Whether to shuffle training data (handled by sampler)
        
        ### Monitoring and Evaluation
        - val_size: Number of validation samples for evaluation
        - val_interval: Steps between validation evaluations
        - checkpoint_interval: Steps between checkpoint saves
        
        ### Optimization Features
        - autocast: Enable automatic mixed precision training
        - **kwargs: Additional strategy-specific or trainer-specific parameters
        
        ## Critical Implementation Details
        
        ### Memory Safety
        - Original model is deep-copied to CPU before multiprocessing
        - Prevents GPU memory sharing conflicts between processes
        - Final model tensors are moved back to CPU for safe aggregation
        
        ### Process Isolation
        - Each worker process has completely independent training state
        - No shared memory between workers except for distributed communication
        - Worker failures are isolated and don't crash the main process
        
        ### Port Management
        - Automatically increments self.port to avoid conflicts
        - Essential for running multiple training jobs sequentially
        - Uses TrainingConstants.DEFAULT_MASTER_PORT as base
        
        ### Validation Logic
        - Ensures val_size ≥ batch_size for meaningful evaluation
        - Validates that effective batch size is achievable with minibatch size
        - Checks that training configuration is internally consistent
        
        ## Error Conditions
        
        ### Configuration Errors
        - val_size < batch_size: Insufficient validation data
        - Invalid device specification: Unsupported hardware
        - Strategy incompatibility: Strategy doesn't support num_nodes
        
        ### Runtime Errors
        - Worker process failures: Individual workers crash during training
        - Communication failures: Distributed operations fail
        - Out of memory: GPU memory exhaustion in worker processes
        
        ## Return Value
        
        Returns a PyTorch model containing the averaged parameters from all
        worker processes. This model represents the final trained state and
        can be used for inference or further training.
        
        Returns None if training fails completely (all workers fail).
        
        Args:
            num_epochs: Number of complete passes through the training dataset
            strategy: Distributed communication strategy for training coordination
            num_nodes: Number of distributed worker processes to spawn
            max_steps: Optional limit on total training steps (overrides num_epochs)
            device: Target device type ("cuda", "mps", "cpu", None for auto-detect)
            devices: Specific GPU device IDs for multi-GPU training
            batch_size: Effective batch size after gradient accumulation
            minibatch_size: Per-forward-pass batch size for memory management
            shuffle: Whether to shuffle training data (handled by distributed sampler)
            val_size: Number of validation samples for evaluation
            val_interval: Number of training steps between validation evaluations
            autocast: Enable automatic mixed precision training
            checkpoint_interval: Number of steps between checkpoint saves
            **kwargs: Additional parameters passed to trainer and strategy
            
        Returns:
            torch.nn.Module: Trained model with averaged parameters from all workers,
                            or None if training fails completely
            
        Raises:
            AssertionError: If val_size < batch_size (insufficient validation data)
            RuntimeError: If multiprocessing fails or workers crash
            
        Called by:
            User training scripts and example code
            
        Calls:
            torch.multiprocessing.spawn() for worker process management
            _average_model_states() for final model aggregation
        """
        # Store parameters
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.device = device
        self.devices = devices
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        self.val_size = val_size
        self.val_interval = val_interval
        self.autocast = autocast
        self.checkpoint_interval = checkpoint_interval

        assert self.val_size // self.batch_size > 0, "val_size must be geq batch_size"

        if hasattr(self, "kwargs"):
            self.kwargs.update(kwargs)
        else:
            self.kwargs = kwargs

        self.port += 1

        # Move a *copy* of the model to CPU so that pickling for mp.spawn does not attempt to share GPU storage.
        cpu_model = copy.deepcopy(self.model_orig).cpu()

        config = TrainingConfig(
            model=cpu_model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            strategy=strategy,
            num_epochs=num_epochs,
            num_nodes=num_nodes,
            max_steps=max_steps,
            device=device,
            devices=devices,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            shuffle=shuffle,
            val_size=val_size,
            val_interval=val_interval,
            autocast=autocast,
            checkpoint_interval=checkpoint_interval,
            trainer_class=self.__class__,
            kwargs=self.kwargs,
        )
        
        # Create a manager and queue for collecting results
        manager = mp.Manager()
        result_queue = manager.Queue()

        # Use mp.spawn with the result queue
        mp.spawn(
            _worker,
            args=(config, result_queue),
            nprocs=config.num_nodes,
            start_method="spawn",
            join=True,
        )

        # Collect results
        model_states = {}
        for _ in range(config.num_nodes):
            rank, state_dict = result_queue.get()
            model_states[rank] = state_dict

        # Average the models
        averaged_state_dict = _average_model_states(model_states)

        if averaged_state_dict is not None:
            # Create a copy of the original model and load the averaged state
            final_model = copy.deepcopy(self.model_orig)
            final_model.load_state_dict(averaged_state_dict)
            return final_model
        else:
            return None

    def _fit_process(self, rank):
        """
        The core training logic that runs in each process.
        Renamed from _fit and removed the spawn call.
        Returns the final model state dict.
        """
        self.rank = rank

        self._build_connection()

        self.model = copy.deepcopy(self.model_orig).to(self.device)

        self.strategy = copy.deepcopy(self.strategy)
        self.strategy._init_node(self.model, self.rank, self.num_nodes)

        # Handle dataset factory vs direct dataset for sampler creation
        if callable(self.train_dataset):
            # For dataset factory, we don't need a distributed sampler
            # since the factory should return the appropriate subset for this rank
            self.sampler = None
        else:
            # For direct dataset, use DistributedSampler as before
            self.sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.num_nodes,
                rank=self.rank,
                shuffle=self.shuffle,
            )

        sim = TrainNode(
            self.model,
            self.train_dataset,
            self.sampler,
            self.val_dataset,
            self.strategy,
            self.device,
            self.rank,
            self.num_nodes,
            num_epochs=self.num_epochs,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            val_size=self.val_size,
            val_interval=self.val_interval,
            checkpoint_interval=self.checkpoint_interval,
            autocast=self.autocast,
            **self.kwargs,
        )

        final_state_dict = sim.train()

        self._process_cleanup()

        return final_state_dict

    @abstractmethod
    def _build_connection(self):
        raise NotImplementedError

    def _process_cleanup(self):
        dist.destroy_process_group()


class LocalTrainer(Trainer):
    def _build_connection(self):
        """
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        if self.device == "" or self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # initialize the process group
        if self.device == "cuda":
            # If we haven't specified devices, use all devices.
            if self.devices is None:
                self.devices = range(torch.cuda.device_count())

            dist.init_process_group(
                "nccl" if len(self.devices) == self.num_nodes else "gloo",
                rank=self.rank,
                world_size=self.num_nodes,
            )
            self.device = torch.device(
                f"cuda:{self.devices[self.rank % len(self.devices)]}"
            )
            torch.cuda.set_device(self.device)
        elif self.device == "cpu":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("cpu")
        elif self.device == "mps":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device type: {self.device}")

        print(f"Rank {self.rank} using device {self.device}")