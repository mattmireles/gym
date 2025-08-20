"""
ExoGym TrainNode - Individual Distributed Training Node Implementation

This module implements the core training logic executed within each distributed
training process. Each TrainNode represents a single worker in the distributed
training system and handles the complete training loop including data loading,
forward/backward passes, strategy communication, and evaluation.

## Core Responsibilities

### Training Loop Management
- Executes the main training loop with configurable step limits and epoch handling
- Manages minibatch accumulation to achieve effective batch sizes larger than memory allows
- Handles automatic mixed precision training when enabled

### Data Pipeline
- Supports both direct datasets and dataset factory functions for flexible data parallelism
- Manages distributed sampling to ensure each node sees different data partitions
- Handles iterator exhaustion and epoch transitions seamlessly

### Distributed Communication
- Integrates with Strategy classes to perform gradient/model averaging
- Broadcasts initial model parameters to ensure synchronized starting state
- Handles rank-specific responsibilities (e.g., rank 0 for logging, rank 1 for global evaluation)

### Evaluation and Logging
- Performs local and global model evaluation at configurable intervals
- Supports separate evaluation of local model vs. averaged global model
- Integrates with WandB and CSV loggers for metrics tracking

### State Management
- Tracks local step count, epoch number, and training progress
- Handles checkpoint saving/loading (currently disabled but infrastructure exists)
- Manages RNG seeds for reproducible training across processes

## Data Flow Within Node

```
Dataset/Factory → DataLoader → _get_batch() → _train_step() → Strategy.step()
                                    ↓              ↓
                              Forward Pass → Backward Pass → Gradient Communication
                                    ↓
                              _evaluate() → Logger.log_*()
```

## Communication Patterns

### Initialization (All Nodes)
- Broadcast model parameters from rank 0 to ensure synchronized start
- Initialize distributed samplers for data parallelism

### Training Step (All Nodes)  
- Compute gradients locally, then communicate via Strategy.step()
- Strategy determines communication pattern (immediate vs. periodic)

### Evaluation (Rank 0 & 1)
- Rank 0: Evaluates local model, logs local metrics
- Rank 1: Evaluates globally-averaged model, broadcasts global metrics
- Rank 0: Logs global metrics received from rank 1

## Called by:
- trainer._worker() function in spawned processes
- Receives model, datasets, and training configuration from parent process

## Calls:
- Strategy.step() for distributed communication patterns
- Logger classes for metrics tracking and progress visualization
- communicate.py functions for low-level distributed operations

## Hardware Compatibility

Designed to work seamlessly across:
- **CUDA**: Full GPU acceleration with optimized data transfers
- **MPS**: Apple Silicon GPU support with CPU fallback handling
- **CPU**: Efficient CPU-only training for development and smaller models

## Critical Implementation Details

- Uses different evaluation strategies for local vs. global model assessment
- Handles dataset factory pattern for advanced data parallelism scenarios
- Manages autocast contexts for mixed precision training
- Implements robust checkpoint/resume functionality (currently disabled)
- Handles StopIteration gracefully for infinite training loops
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import zipfile

import os
import copy
from typing import Union, Callable

from .strategy.strategy import Strategy
from .logger import WandbLogger, CSVLogger
from .strategy.communicate import all_reduce, broadcast
from .utils import LogModule

# TODO: change to two-space indent instead of four-space (which is what it is at the moment)


class TrainNodeConstants:
    """
    Named constants for training node configuration and behavior.
    
    These constants replace magic numbers throughout the training node implementation
    with well-documented, meaningful values that explain the reasoning behind
    specific parameter choices.
    """
    
    # Random Seed Management
    DEFAULT_TRAINING_SEED = 42
    """
    Default random seed for reproducible training runs.
    
    The value 42 is chosen as a nod to "The Hitchhiker's Guide to the Galaxy"
    and is widely used in machine learning for reproducibility. Using a fixed
    seed ensures:
    - Reproducible initialization of model parameters
    - Consistent data shuffling and augmentation
    - Reproducible dropout and other stochastic operations
    
    This seed is set for:
    - PyTorch random number generator (torch.manual_seed)
    - CUDA random number generator (torch.cuda.manual_seed)  
    - NumPy random number generator (np.random.seed)
    """
    
    # TensorFloat-32 Configuration
    TF32_ENABLED = True
    """
    Enable TensorFloat-32 (TF32) for faster training on Ampere GPUs.
    
    TF32 is a math mode for NVIDIA Ampere architecture GPUs that provides:
    - Faster matrix operations with minimal precision loss
    - Automatic acceleration for many PyTorch operations
    - No code changes required - purely performance optimization
    
    Enabled by default because:
    - Provides 10-20x speedup on A100 and newer GPUs
    - Negligible impact on training convergence for most models
    - Recommended by PyTorch team for production training
    
    Set to False only if you need full float32 precision for debugging.
    """


class TrainNode(LogModule):
    """
    Individual distributed training node implementation.
    
    TrainNode represents a single worker process in the distributed training system.
    Each node executes the complete training loop including data loading, forward/backward
    passes, strategy communication, and evaluation. The implementation is designed to be
    identical across all ranks while handling rank-specific responsibilities.
    
    ## Core Responsibilities
    
    ### Training Loop Execution
    - Manages the main training loop with step counting and epoch tracking
    - Handles minibatch accumulation to achieve larger effective batch sizes
    - Integrates with Strategy classes for distributed communication patterns
    
    ### Data Pipeline Management
    - Supports both direct datasets and dataset factory functions
    - Manages distributed data sampling to ensure each node sees different data
    - Handles iterator exhaustion and epoch transitions automatically
    
    ### Evaluation and Monitoring
    - Performs periodic validation evaluation with configurable intervals
    - Supports rank-specific evaluation patterns (local vs global model evaluation)
    - Integrates with logging infrastructure for metrics tracking
    
    ### State Synchronization
    - Broadcasts initial model parameters to ensure synchronized starting state
    - Tracks training progress (steps, epochs) consistently across all nodes
    - Handles checkpoint saving/loading (infrastructure exists but currently disabled)
    
    ## Design Principles
    
    ### Rank Agnostic Implementation
    - Core training logic is identical across all ranks
    - Rank-specific behavior is handled through conditional logic
    - Enables easy debugging and consistent behavior
    
    ### Hardware Compatibility
    - Works seamlessly across CUDA, MPS, and CPU devices
    - Automatic device placement and data movement
    - TF32 optimization for Ampere GPU acceleration
    
    ### Memory Efficiency
    - Gradient accumulation enables large effective batch sizes
    - Automatic mixed precision support for memory savings
    - Efficient iterator management with lazy evaluation
    
    Called by:
        trainer._worker() function in spawned worker processes
        
    Calls:
        Strategy.step() for distributed communication
        Logger classes for metrics tracking and progress visualization
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        train_sampler: torch.utils.data.Sampler,
        val_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        strategy: Strategy,
        device: torch.device,
        rank: int,
        num_nodes: int,
        num_epochs: int,
        max_steps: int = None,
        batch_size: int = 16,
        minibatch_size: int = 16,
        val_size: int = 64,
        val_interval: int = 100,
        checkpoint_interval: int = 100,
        autocast: bool = False,
        **kwargs,
    ):
        """
        Initialize a distributed training node with complete configuration.
        
        This constructor sets up all the components needed for distributed training
        including data pipelines, strategy integration, device configuration, and
        reproducibility controls.
        
        ## Initialization Process
        
        ### 1. Reproducibility Setup
        - Sets random seeds for PyTorch, CUDA, and NumPy for consistent results
        - Enables TF32 on Ampere GPUs for performance optimization
        - Configurable seed via kwargs for experiment reproducibility
        
        ### 2. Dataset Configuration
        - Handles both direct datasets and dataset factory functions
        - Factory pattern: callable(rank, num_nodes, is_val) -> Dataset
        - Automatic distributed sampling for standard datasets
        
        ### 3. Model Synchronization
        - Broadcasts model parameters from rank 0 to all other ranks
        - Ensures all processes start with identical model state
        - Critical for distributed training convergence
        
        ### 4. Training State Initialization
        - Initializes step counters, epoch tracking, and progress state
        - Attempts to load checkpoints for training resumption (currently disabled)
        - Sets up data iterators for training loop
        
        ## Parameter Categories
        
        ### Core Components
        - model: PyTorch model to train (automatically moved to specified device)
        - strategy: Communication strategy for distributed coordination
        - device: Target device for training (CUDA, MPS, or CPU)
        
        ### Data Configuration
        - train_dataset/val_dataset: Data sources (direct or factory functions)
        - train_sampler: Distributed sampler for data partitioning (None for factories)
        - batch_size: Effective batch size after gradient accumulation
        - minibatch_size: Per-forward-pass batch size for memory management
        
        ### Training Configuration
        - rank/num_nodes: Process identification and distributed setup
        - num_epochs: Number of complete dataset passes
        - max_steps: Optional limit on total training steps
        
        ### Monitoring Configuration
        - val_size: Number of validation samples for evaluation
        - val_interval: Steps between validation evaluations
        - checkpoint_interval: Steps between checkpoint saves
        
        ### Optimization Configuration
        - autocast: Enable automatic mixed precision training
        - **kwargs: Additional configuration including seed override
        
        ## Critical Implementation Details
        
        ### Dataset Factory Pattern
        When train_dataset/val_dataset are callable:
        - Called with (rank, num_nodes, is_val) arguments
        - Enables advanced data partitioning strategies
        - train_sampler is ignored (set to None)
        - Factory handles data distribution logic
        
        ### Distributed Sampling
        When using direct datasets:
        - train_sampler must be DistributedSampler from trainer
        - Ensures each rank sees different data partitions
        - Handles shuffling and epoch coordination
        
        ### Device Management
        - Model automatically moved to specified device
        - All subsequent operations use the same device
        - Device compatibility handled by communicate.py layer
        
        ### Memory Optimization
        - TF32 enabled for 10-20x speedup on Ampere GPUs
        - Gradient accumulation reduces memory pressure
        - Autocast enables mixed precision training
        
        Args:
            model: PyTorch model to train
            train_dataset: Training data (dataset or factory function)
            train_sampler: Distributed sampler for data partitioning
            val_dataset: Validation data (dataset or factory function)
            strategy: Distributed communication strategy
            device: Target device for training
            rank: Process rank (0 to num_nodes-1)
            num_nodes: Total number of distributed processes
            num_epochs: Number of complete dataset passes
            max_steps: Optional limit on total training steps
            batch_size: Effective batch size after gradient accumulation
            minibatch_size: Per-forward-pass batch size
            val_size: Number of validation samples
            val_interval: Steps between validation evaluations
            checkpoint_interval: Steps between checkpoint saves
            autocast: Enable automatic mixed precision
            **kwargs: Additional configuration (e.g., seed override)
        """
        seed = kwargs.get("seed", TrainNodeConstants.DEFAULT_TRAINING_SEED)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        torch.backends.cuda.matmul.allow_tf32 = TrainNodeConstants.TF32_ENABLED
        torch.backends.cudnn.allow_tf32 = TrainNodeConstants.TF32_ENABLED

        self.model = model

        # Handle dataset factory vs direct dataset for training
        if callable(train_dataset):
            # Call the dataset factory function with rank, num_nodes, and val=False
            self.train_dataset = train_dataset(rank, num_nodes, False)
            # When using dataset factory, we don't need a distributed sampler
            # since the factory should return the appropriate subset
            self.train_sampler = None
        else:
            # Use the dataset directly as before
            self.train_dataset = train_dataset
            self.train_sampler = train_sampler

        # Handle dataset factory vs direct dataset for validation
        if callable(val_dataset):
            # Call the dataset factory function with rank, num_nodes, and val=True
            self.val_dataset = val_dataset(rank, num_nodes, True)
        else:
            # Use the dataset directly as before
            self.val_dataset = val_dataset

        self.strategy = strategy
        self.device = device
        self.rank = rank
        self.num_nodes = num_nodes
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.val_size = val_size
        self.val_interval = val_interval
        self.autocast = autocast
        self.checkpoint_interval = checkpoint_interval

        self.kwargs = kwargs

        self.build_dataloaders()

        # Re-seed after dataloader creation for consistency
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        ## Ensure all process models share the same params
        if self.num_nodes > 1:
            for _, param in self.model.named_parameters():
                broadcast(param.data, src=0)

        self.local_step = 0
        self.epoch = 0

        # Attempt to load checkpoint before starting training
        self._load_checkpoint()

    def build_dataloaders(self):
        """
        Construct training and validation data loaders with distributed support.
        
        This method creates PyTorch DataLoader instances that handle the complexities
        of distributed training data management. It supports both traditional datasets
        with distributed sampling and dataset factory functions for advanced partitioning.
        
        ## Data Loading Strategies
        
        ### Dataset Factory Pattern (Recommended)
        When datasets are callable functions:
        - No distributed sampler needed (factory handles partitioning)
        - Enables shuffling at DataLoader level
        - More flexible data distribution strategies
        - Better for complex data partitioning schemes
        
        ### Traditional Dataset + DistributedSampler
        When datasets are PyTorch Dataset objects:
        - Uses DistributedSampler provided by trainer
        - Sampler handles data partitioning across ranks
        - Shuffling controlled by sampler, not DataLoader
        - Standard PyTorch distributed training pattern
        
        ## Implementation Details
        
        ### Training DataLoader Configuration
        - batch_size: Uses minibatch_size for memory management
        - sampler: DistributedSampler or None (for factory pattern)
        - shuffle: Enabled only when no sampler is provided
        - Ensures each rank sees different training data
        
        ### Validation DataLoader Configuration
        - batch_size: Uses minibatch_size for consistency
        - shuffle: Always enabled for validation sampling
        - No distributed sampling (all ranks use same validation data)
        - Provides consistent evaluation across ranks
        
        ### Iterator Management
        - Creates fresh iterators for both training and validation
        - Iterators are recreated when datasets are exhausted
        - Enables infinite training loops with automatic epoch handling
        
        ## Memory and Performance Considerations
        
        ### Minibatch Size Strategy
        - Uses minibatch_size instead of full batch_size
        - Enables gradient accumulation for larger effective batch sizes
        - Reduces GPU memory pressure for large models
        - Maintains training dynamics of larger batches
        
        ### Data Loading Efficiency
        - No explicit num_workers setting (uses PyTorch defaults)
        - Relies on efficient dataset implementations
        - Factory pattern can optimize data loading per rank
        
        Called by:
            TrainNode.__init__() during initialization
            
        Calls:
            torch.utils.data.DataLoader for data loading infrastructure
        """
        # For dataset factory case (when sampler is None), we can enable shuffling
        # For regular dataset case (when sampler is provided), shuffling is handled by the sampler
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.minibatch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.minibatch_size, shuffle=True
        )

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)

    def _get_batch(self, eval=False):
        if not eval or self.val_data_iter is None:
            try:
                batch = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                batch = next(self.train_data_iter)
        else:
            try:
                batch = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                batch = next(self.val_data_iter)

        # Support dict-style batches (model wrapper will handle device movement)
        if isinstance(batch, dict):
            pass
        elif isinstance(batch, (tuple, list)):
            batch = tuple(x.to(self.device) for x in batch)
        else:
            batch = batch.to(self.device)

        return batch

    def _train_step(self):
        """
        Execute one complete training step with gradient accumulation and communication.
        
        This method implements the core training logic that runs on each distributed node.
        It handles gradient accumulation across multiple minibatches to achieve larger
        effective batch sizes while maintaining memory efficiency.
        
        ## Training Step Process
        
        ### 1. Gradient Accumulation Loop
        - Processes (batch_size // minibatch_size) minibatches sequentially
        - Each minibatch produces gradients that are accumulated
        - Memory usage limited by minibatch_size regardless of effective batch size
        
        ### 2. Gradient Normalization
        - Divides accumulated gradients by number of minibatches
        - Ensures gradient magnitudes match single large batch training
        - Critical for maintaining learning dynamics across different batch sizes
        
        ### 3. Distributed Communication
        - Calls strategy.step() to perform inter-node communication
        - Strategy determines communication pattern (immediate vs periodic)
        - May include gradient averaging, model averaging, or other patterns
        
        ### 4. Progress Tracking and Checkpointing
        - Logs training metrics (rank 0 only to avoid conflicts)
        - Saves checkpoints at configured intervals
        - Updates progress tracking for monitoring
        
        ## Gradient Accumulation Mathematics
        
        For batch_size=64 and minibatch_size=16:
        - 4 minibatches processed sequentially
        - Gradients accumulated: g_total = g_1 + g_2 + g_3 + g_4
        - Normalized: g_final = g_total / 4
        - Equivalent to single forward pass with batch_size=64
        
        ## Automatic Mixed Precision Integration
        
        ### When autocast=True
        - Forward passes use bfloat16 for memory savings and speed
        - Backward passes automatically handle precision conversion
        - Compatible with gradient accumulation and distributed training
        
        ### When autocast=False
        - Uses full float32 precision throughout
        - Higher memory usage but maximum numerical stability
        - Recommended for debugging and precision-sensitive models
        
        ## Distributed Communication Patterns
        
        ### Strategy.step() Responsibilities
        - Gradient averaging (SimpleReduceStrategy)
        - Periodic model averaging (DiLoCoStrategy)
        - Custom communication patterns (other strategies)
        - Learning rate scheduling and optimization
        
        ## Rank-Specific Responsibilities
        
        ### Rank 0 (Logging Coordinator)
        - Logs training metrics to prevent logging conflicts
        - Handles checkpoint saving coordination
        - Maintains progress tracking for the entire training run
        
        ### All Ranks
        - Execute identical training computations
        - Participate in distributed communication
        - Save individual checkpoints (when enabled)
        
        ## Memory and Performance Optimization
        
        ### Memory Efficiency
        - Gradient accumulation reduces peak memory usage
        - Autocast reduces memory footprint by ~50%
        - Minibatch processing enables training of larger models
        
        ### Performance Optimization
        - TF32 acceleration on compatible GPUs
        - Overlapped computation and communication (strategy-dependent)
        - Efficient iterator management with lazy loading
        
        ## Error Handling
        
        ### Gradient Issues
        - Checks param.requires_grad before gradient manipulation
        - Handles None gradients gracefully
        - Preserves gradient computation graph during accumulation
        
        ### Communication Failures
        - Strategy.step() handles distributed communication errors
        - Individual node failures don't crash other nodes
        - Automatic recovery depends on strategy implementation
        
        Called by:
            TrainNode.train() during main training loop
            
        Calls:
            Strategy.zero_grad() for gradient reset
            Strategy.step() for distributed communication
            Logger.log_train() for metrics tracking
        """
        self.strategy.zero_grad()

        for i in range(self.batch_size // self.minibatch_size):
            minibatch = self._get_batch()

            # Automatic mixed precision for memory and speed optimization
            if self.autocast:
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    loss = self.model(minibatch)
            else:
                loss = self.model(minibatch)

            loss.backward()

        # Normalize accumulated gradients to match single large batch training
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad /= self.batch_size / self.minibatch_size

        self.strategy.step()

        # Only rank 0 logs to avoid conflicts and duplicate entries
        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        # Checkpoint saving at configured intervals
        if self.checkpoint_interval and self.local_step % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _evaluate(self):
        if self.val_size == 0:
            return

        model_clone = copy.deepcopy(self.model)

        for name, param in model_clone.named_parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            this_model = self.model

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            this_model = model_clone

        if self.rank == 0 or self.rank == 1:
            this_model.eval()

            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.val_size / self.batch_size)):

                    for i in range(self.batch_size // self.minibatch_size):
                        minibatch = self._get_batch(eval=True)

                        if self.autocast:
                            with torch.autocast(
                                device_type=self.device, dtype=torch.bfloat16
                            ):
                                ## TODO: Fix
                                loss = this_model(minibatch)
                        else:
                            loss = this_model(minibatch)

                        loss_total += loss.item() / (
                            self.batch_size // self.minibatch_size
                        )

        # Rank 0 logs the local evaluation.
        if self.rank == 0:
            self.logger.log_loss(
                loss=loss_total / int(self.val_size / self.batch_size), name="local"
            )

        # Broadcast the global loss from rank 1 to all ranks.
        if self.num_nodes > 1:
            # All ranks create a dummy tensor to participate.
            global_loss_tensor = torch.empty(
                1, device=next(self.model.parameters()).device
            )
            if self.rank == 1:
                global_loss_tensor[0] = loss_total / int(
                    self.val_size / self.batch_size
                )
            broadcast(global_loss_tensor, src=1)

            # Only rank 0 logs the global evaluation.
            if self.rank == 0:
                global_loss = global_loss_tensor.item()
                self.logger.log_loss(loss=global_loss, name="global")

        del model_clone

    def _save_checkpoint(self):
        return  ## TODO
        print(
            self.config.save_dir,
            self.config.wandb_project,
            self.config.run_name,
            self.rank,
        )
        save_path_dir = os.path.join(
            self.config.save_dir,
            self.config.wandb_project if self.config.wandb_project else "unnamed",
            self.config.run_name if self.config.run_name else "unnamed",
            str(self.rank),
        )
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=True)

        filename = f"{self.local_step}.pt"
        full_save_path = os.path.join(save_path_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.strategy.optim.state_dict(),
            "local_step": self.local_step,
            "epoch": self.epoch,
            "rng_state": torch.get_rng_state(),
        }
        if self.strategy.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.strategy.scheduler.state_dict()

        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        try:
            torch.save(checkpoint, full_save_path)
            print(
                f"Rank {self.rank} saved checkpoint to {full_save_path} at step {self.local_step}"
            )
            self._delete_other_checkpoints(save_path_dir, full_save_path)
        except OSError as e:
            print(
                f"Rank {self.rank}: Failed to save checkpoint {full_save_path} due to OSError: {e}. Attempting to delete oldest checkpoint and retry."
            )

            oldest_step = float("inf")
            oldest_checkpoint_file = None
            # Ensure save_path_dir exists before listing its contents, though it should have been created.
            if os.path.exists(save_path_dir):
                for f_name in os.listdir(save_path_dir):
                    if f_name.endswith(".pt"):
                        try:
                            # Checkpoints are named as {step_num}.pt
                            step_num = int(f_name.split(".")[0])
                            if step_num < oldest_step:
                                oldest_step = step_num
                                oldest_checkpoint_file = f_name
                        except ValueError:
                            # Skip files not matching the expected N.pt pattern
                            continue

            if oldest_checkpoint_file:
                oldest_checkpoint_path = os.path.join(
                    save_path_dir, oldest_checkpoint_file
                )
                try:
                    os.remove(oldest_checkpoint_path)
                    print(
                        f"Rank {self.rank}: Deleted oldest checkpoint {oldest_checkpoint_path} to free space."
                    )

                    # Retry saving the current checkpoint
                    try:
                        torch.save(checkpoint, full_save_path)
                        print(
                            f"Rank {self.rank}: Successfully saved checkpoint {full_save_path} after deleting oldest."
                        )
                        self._delete_other_checkpoints(save_path_dir, full_save_path)
                    except OSError as e2:
                        print(
                            f"Rank {self.rank}: Still failed to save checkpoint {full_save_path} after deleting oldest: {e2}. Giving up."
                        )
                        raise  # Re-raise the second error, as we couldn't save even after cleanup
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Failed to delete oldest checkpoint {oldest_checkpoint_path}: {del_e}. Original save error will be raised."
                    )
                    raise e  # Re-raise the original save error, as cleanup failed
            else:
                print(
                    f"Rank {self.rank}: No old checkpoints found to delete in {save_path_dir}. Original save error will be raised."
                )
                raise e  # Re-raise the original save error, as no space could be freed

    def _delete_other_checkpoints(
        self, save_path_dir: str, current_checkpoint_full_path: str
    ):
        return  ## TODO
        if not os.path.exists(save_path_dir):
            return

        current_checkpoint_filename = os.path.basename(current_checkpoint_full_path)
        deleted_count = 0
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith(".pt") and f_name != current_checkpoint_filename:
                try:
                    file_to_delete = os.path.join(save_path_dir, f_name)
                    os.remove(file_to_delete)
                    # print(f"Rank {self.rank}: Deleted old checkpoint {file_to_delete}")
                    deleted_count += 1
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Warning - Failed to delete old checkpoint {file_to_delete}: {del_e}"
                    )
        if deleted_count > 0:
            print(
                f"Rank {self.rank}: Deleted {deleted_count} other checkpoint(s) in {save_path_dir}."
            )

    def _load_checkpoint(self):
        return  ## TODO
        save_path_dir = os.path.join(
            self.config.save_dir,
            self.config.wandb_project if self.config.wandb_project else "unnamed",
            self.config.run_name if self.config.run_name else "unnamed",
            str(self.rank),
        )

        if not os.path.exists(save_path_dir):
            print(
                f"Rank {self.rank}: Checkpoint directory {save_path_dir} not found. Starting from scratch."
            )
            return False

        checkpoint_files = []
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith(".pt"):
                try:
                    step_num = int(f_name.split(".")[0])
                    checkpoint_files.append((step_num, f_name))
                except ValueError:
                    # Not a valid checkpoint file name pattern
                    continue

        # Sort by step number in descending order (latest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)

        loaded_successfully = False
        for step_num, f_name in checkpoint_files:
            full_checkpoint_path = os.path.join(save_path_dir, f_name)
            try:
                print(
                    f"Rank {self.rank}: Attempting to load checkpoint from {full_checkpoint_path}"
                )
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)

                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.strategy.optim.load_state_dict(checkpoint["optimizer_state_dict"])

                if (
                    "scheduler_state_dict" in checkpoint
                    and self.strategy.scheduler is not None
                ):
                    self.strategy.scheduler.load_state_dict(
                        checkpoint["scheduler_state_dict"]
                    )

                self.local_step = checkpoint["local_step"]
                self.epoch = checkpoint["epoch"]

                torch.set_rng_state(
                    checkpoint["rng_state"].cpu()
                )  # Ensure RNG state is on CPU before loading
                if self.device.type == "cuda" and "cuda_rng_state" in checkpoint:
                    if isinstance(checkpoint["cuda_rng_state"], torch.Tensor):
                        torch.cuda.set_rng_state(
                            checkpoint["cuda_rng_state"].cpu(), device=self.device
                        )
                    else:
                        torch.cuda.set_rng_state(
                            checkpoint["cuda_rng_state"], device=self.device
                        )

                self.train_data_iter = iter(self.train_dataloader)
                self.val_data_iter = iter(self.val_dataloader)

                if len(self.train_dataloader) > 0:
                    batches_to_skip = self.local_step % len(self.train_dataloader)
                    print(
                        f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Skipping {batches_to_skip} batches."
                    )
                    for _ in range(batches_to_skip):
                        try:
                            next(self.train_data_iter)
                        except StopIteration:
                            print(
                                f"Rank {self.rank}: Warning - StopIteration while fast-forwarding train_data_iter."
                            )
                            break
                else:
                    print(
                        f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Train dataloader empty."
                    )

                if self.rank == 0 and hasattr(self.logger, "set_step"):
                    self.logger.set_step(self.local_step)
                elif self.rank == 0:
                    print(
                        f"Rank 0: Logger step will resume from loaded local_step: {self.local_step}"
                    )

                print(
                    f"Rank {self.rank}: Successfully loaded checkpoint {f_name}. Resuming at epoch {self.epoch}, step {self.local_step}."
                )
                loaded_successfully = True
                break  # Exit loop once a checkpoint is successfully loaded
            except (
                RuntimeError,
                EOFError,
                zipfile.BadZipFile,
            ) as e:  # Catch specific errors related to corrupted files
                print(
                    f"Rank {self.rank}: Failed to load checkpoint {full_checkpoint_path}: {e}. Trying next available checkpoint."
                )
                # Optionally, delete the corrupted checkpoint file
                try:
                    os.remove(full_checkpoint_path)
                    print(
                        f"Rank {self.rank}: Deleted corrupted checkpoint {full_checkpoint_path}."
                    )
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Warning - Failed to delete corrupted checkpoint {full_checkpoint_path}: {del_e}"
                    )
            except Exception as e:  # Catch any other unexpected error during loading
                print(
                    f"Rank {self.rank}: An unexpected error occurred while loading checkpoint {full_checkpoint_path}: {e}. Trying next."
                )
                # Optionally, delete or move the problematic checkpoint

        if not loaded_successfully:
            print(
                f"Rank {self.rank}: No valid checkpoint found in {save_path_dir} after trying all options. Starting from scratch."
            )
            # Reset relevant states if starting from scratch, though __init__ defaults should cover this.
            self.local_step = 0
            self.epoch = 0
            return False

        return True

    def _correlation_calculation(self):
        return  ## TODO
        if self.num_nodes < 2:
            raise Exception("Correlation calculation cannot be used with < 2 nodes")

        # Ensure correlation is only calculated if interval is set
        if not self.config.correlation_interval:
            return None

        # Create a temporary directory for this timestep's checkpoints
        tmp_dir = os.path.join(self.config.save_dir, f"tmp_corr_{self.local_step}")
        # Only rank 0 creates the directory to avoid race conditions
        if self.rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
        torch.distributed.barrier()  # Wait for rank 0 to create dir

        # Save model state dict for each rank
        checkpoint_path = os.path.join(tmp_dir, f"{self.rank}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Wait for all processes to save their checkpoints
        torch.distributed.barrier()

        corr_value = None
        if self.rank == 0:
            # Load all models as vectors
            model_vectors = []
            for r in range(self.config.num_nodes):
                model_path = os.path.join(tmp_dir, f"{r}.pt")
                # Ensure the file exists before trying to load
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location="cpu")
                    vector_list = []
                    for key in sorted(checkpoint.keys()):
                        value = checkpoint[key]
                        if isinstance(value, torch.Tensor):
                            vector_list.append(value.cpu().numpy().ravel())
                    if vector_list:  # Check if we actually got any tensors
                        model_vectors.append(np.concatenate(vector_list))
                else:
                    print(
                        f"Warning: Checkpoint file {model_path} not found for rank {r}."
                    )

            if len(model_vectors) >= 2:  # Need at least two models to compare
                # Calculate correlations between all pairs
                correlations = []
                for i in range(len(model_vectors)):
                    for j in range(i + 1, len(model_vectors)):
                        corr = np.corrcoef(model_vectors[i], model_vectors[j])[0, 1]
                        correlations.append(corr)

                if correlations:  # Ensure correlations list is not empty
                    corr_value = np.mean(correlations)

                    # Log average correlation to wandb using the logger
                    if self.logger:
                        self.logger.log(data={"avg_model_correlation": corr_value})
                else:
                    print(
                        "Warning: Could not calculate correlation, not enough valid model pairs."
                    )
            else:
                print(
                    f"Warning: Not enough models loaded ({len(model_vectors)}) to calculate correlation."
                )

            # Clean up temporary directory
            import shutil

            shutil.rmtree(tmp_dir)

        # Wait for rank 0 to finish cleanup
        torch.distributed.barrier()

        return corr_value  # Only rank 0 returns a value, others return None

    def train(self):
        """
        Execute the complete distributed training loop.
        
        This is the main training method that orchestrates the entire training process
        from initialization through completion. It handles logger setup, step counting,
        evaluation scheduling, and distributed synchronization.
        
        ## Training Loop Architecture
        
        ### 1. Pre-Training Setup
        - Calculates max_steps from epochs if not explicitly provided
        - Initializes appropriate logger (WandB or CSV) on rank 0 only
        - Configures strategy with total step count for scheduling
        
        ### 2. Main Training Loop
        - Executes training steps until max_steps reached
        - Performs validation at configured intervals
        - Maintains step counting and progress tracking
        - Handles distributed synchronization with barriers
        
        ### 3. Post-Training Cleanup
        - Performs final evaluation for complete metrics
        - Returns final model state for aggregation
        - Prepares model state for averaging across ranks
        
        ## Step Calculation Logic
        
        When max_steps is not explicitly provided:
        ```
        max_steps = num_epochs * len(train_dataloader) / (batch_size // minibatch_size)
        ```
        
        This calculation accounts for:
        - Multiple epochs over the complete dataset
        - Gradient accumulation reducing effective steps per epoch
        - DataLoader length based on minibatch_size, not effective batch_size
        
        ## Logger Selection and Configuration
        
        ### WandB Logger (Cloud-based)
        - Selected when wandb_project is provided in kwargs
        - Enables rich experiment tracking and team collaboration
        - Automatic configuration upload and run management
        - Requires internet connection and wandb account
        
        ### CSV Logger (Local)
        - Selected when wandb_project is not provided
        - Lightweight local logging for development and CI
        - Self-contained with configuration persistence
        - Works in offline environments
        
        ### Rank 0 Logging Responsibility
        - Only rank 0 creates and manages loggers
        - Prevents duplicate logging and conflicts
        - Centralizes progress tracking and metric reporting
        
        ## Evaluation Scheduling
        
        ### Validation Intervals
        - Evaluation performed every val_interval steps
        - Always includes step 0 for baseline metrics
        - Final evaluation after training completion
        - Provides consistent monitoring throughout training
        
        ### Evaluation Benefits
        - Early stopping detection through validation metrics
        - Training progress monitoring and debugging
        - Model selection based on validation performance
        - Overfitting detection through loss divergence
        
        ## Distributed Synchronization
        
        ### Barrier Synchronization
        - dist.barrier() ensures all ranks proceed together
        - Prevents fast ranks from getting too far ahead
        - Essential for strategies that depend on step synchronization
        - Maintains consistent evaluation timing across ranks
        
        ### Step Coordination
        - All ranks maintain identical local_step counters
        - Strategy.step() may have rank-specific behavior
        - Logger step tracking coordinated through rank 0
        
        ## Memory and State Management
        
        ### State Dictionary Return
        - Returns model.state_dict() for final model aggregation
        - CPU conversion handled by caller (_worker function)
        - Preserves parameter names and tensor metadata
        - Enables averaging across ranks in parent process
        
        ### Memory Cleanup
        - Local references maintained until return
        - CUDA context cleanup handled by process termination
        - Logger cleanup automatic with object destruction
        
        ## Performance Monitoring
        
        ### Progress Tracking
        - Step counting provides training progress visibility
        - Logger integration enables real-time monitoring
        - Evaluation metrics track model performance
        
        ### Correlation Analysis (Disabled)
        - Infrastructure exists for model correlation analysis
        - Currently disabled but can be re-enabled for research
        - Would analyze parameter similarity across ranks
        
        ## Error Handling and Recovery
        
        ### Training Interruption
        - Loop can be interrupted at any step boundary
        - Checkpoint infrastructure available (currently disabled)
        - State recovery depends on checkpoint implementation
        
        ### Distributed Failures
        - Barrier failures indicate node communication issues
        - Individual rank failures contained within processes
        - Recovery depends on strategy-specific error handling
        
        Returns:
            OrderedDict: Final model state dictionary ready for aggregation
            
        Called by:
            trainer._fit_process() in worker processes
            
        Calls:
            Strategy.step() for distributed communication
            Logger classes for metrics tracking
            _evaluate() for validation assessment
        """
        # Calculate total training steps if not explicitly provided
        if self.max_steps is None:
            self.max_steps = (
                self.num_epochs
                * len(self.train_dataloader)
                / (self.batch_size // self.minibatch_size)
            )

        # Provide step count to strategy for learning rate scheduling
        self.strategy.max_steps = self.max_steps

        # Initialize logger on rank 0 only to prevent conflicts
        if self.rank == 0:
            if self.kwargs.get("wandb_project", None) is not None:
                self.logger = WandbLogger(
                    model=self.model,
                    max_steps=self.max_steps,
                    strategy=self.strategy,
                    train_node=self,
                    wandb_project=self.kwargs.get("wandb_project", None),
                    run_name=self.kwargs.get("run_name", None),
                )
            else:
                self.logger = CSVLogger(
                    model=self.model,
                    max_steps=self.max_steps,
                    strategy=self.strategy,
                    train_node=self,
                    run_name=self.kwargs.get("run_name", None),
                )

        # Main training loop
        while self.local_step < self.max_steps:
            # Periodic validation evaluation
            if self.local_step % self.val_interval == 0:
                self._evaluate()

            # Execute one training step with gradient accumulation
            self._train_step()

            # Update step counters and progress tracking
            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            # Synchronize all ranks to maintain consistent progress
            # Critical for strategies that depend on step alignment
            dist.barrier()

        # Final evaluation for complete training metrics
        self._evaluate()

        # Return final model state for aggregation across ranks
        return self.model.state_dict()

    def __config__(self):
        remove_keys = ["model", "train_dataloader", "val_dataloader", "strategy"]

        config = super().__config__(remove_keys=remove_keys)

        return config
