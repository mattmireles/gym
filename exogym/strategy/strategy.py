"""
ExoGym Strategy - Base Classes for Distributed Training Communication

This module defines the core Strategy interface and basic implementations for
distributed training communication patterns. Strategies encapsulate how and when
nodes communicate during training, from simple gradient averaging to sophisticated
periodic model averaging schemes.

## Strategy Pattern Architecture

### Strategy (Abstract Base Class)
Defines the interface that all distributed training strategies must implement:
- step(): Called after each training step to perform communication
- zero_grad(): Delegate to underlying optimizer
- Learning rate scheduling and callback management

### SimpleReduceStrategy  
Basic gradient averaging strategy that communicates after every training step:
- Performs all_reduce on gradients immediately after backward pass
- Applies gradient clipping before optimizer step
- Equivalent to traditional data parallel training

## Key Design Principles

### Optimizer Integration
Strategies wrap and manage the underlying optimizers, providing a consistent
interface while adding distributed communication logic. The optimizer is created
in _init_node() to ensure proper device placement.

### Learning Rate Management
Supports sophisticated LR scheduling including:
- Warmup periods with linear scaling
- Cosine annealing with configurable minimum LR
- Callback system for logging LR changes

### Hardware Agnostic Communication
Uses communicate.py abstractions to handle CUDA/MPS/CPU distributed operations
transparently. Strategies don't need to worry about device-specific communication.

## Lifecycle

1. **Construction**: Strategy created with optimizer spec and hyperparameters
2. **Node Initialization**: _init_node() called with model, rank, num_nodes
3. **Training Loop**: step() called after each backward pass
4. **Communication**: Strategy-specific communication pattern executed
5. **Optimization**: Underlying optimizer step performed

## Communication Flow

```
TrainNode._train_step() → Strategy.step() → communicate.all_reduce() → Optimizer.step()
```

## Called by:
- TrainNode during training loop execution
- Receives gradients from backward pass, coordinates with other nodes

## Calls:
- communicate.py functions for distributed operations
- torch.optim optimizers for parameter updates
- LR schedulers for learning rate management

## Implementation Guidelines

When implementing new strategies:
- Inherit from Strategy base class
- Implement step() method with communication logic
- Call super().step() to handle LR scheduling and callbacks
- Use communicate.py functions for hardware-agnostic operations
- Track communication volume with self.nbytes for monitoring

## Current Implementations

- **SimpleReduceStrategy**: Immediate gradient averaging (data parallel equivalent)
- **DiLoCoStrategy**: Periodic model averaging with inner/outer optimizers
- **FederatedAveragingStrategy**: Client-server federated learning pattern
"""

from torch.optim.lr_scheduler import LambdaLR

import math
import torch
import torch.nn.utils as nn_utils

from typing import Dict, Any

from .communicate import all_reduce

from exogym.utils import LogModule

from abc import ABC, abstractmethod

from .optim import OptimSpec, ensure_optim_spec


class StrategyConstants:
    """
    Named constants for strategy configuration and learning rate scheduling.
    
    These constants define default values and limits for distributed training
    strategies, particularly for learning rate scheduling and optimization.
    """
    
    # Learning Rate Scheduling
    MIN_LR_FACTOR = 0.1
    """
    Minimum learning rate factor for cosine annealing schedule.
    
    Represents the minimum LR as a fraction of the base learning rate during
    cosine annealing. Setting to 0.1 means the learning rate will decay to
    10% of its original value at the end of training.
    
    This prevents the learning rate from going to zero, which can cause
    training to stagnate in the final phases. The 0.1 factor provides
    continued learning while maintaining training stability.
    """
    
    # Default Optimizer Parameters
    DEFAULT_SGD_MOMENTUM = 0.9
    """
    Default momentum parameter for SGD optimizers.
    
    Momentum of 0.9 is a widely-used default that provides good convergence
    properties for most optimization scenarios. It helps accelerate gradients
    in the relevant direction and dampens oscillations.
    
    This value is used in strategy examples and default configurations.
    """
    
    DEFAULT_OUTER_LR = 0.7
    """
    Default learning rate for DiLoCo outer optimizer.
    
    Research has shown that outer optimizers in DiLoCo typically require
    higher learning rates than inner optimizers. The value 0.7 has been
    empirically validated for good convergence in DiLoCo training.
    
    This higher LR compensates for the periodic nature of outer optimization
    and the model averaging that occurs every H steps.
    """


class Strategy(ABC, LogModule):
    """
    Abstract base class for all distributed training communication strategies.
    
    Strategy defines the core interface that all distributed training strategies must
    implement. It encapsulates the communication patterns, optimization logic, and
    learning rate scheduling that determine how distributed nodes coordinate during training.
    
    ## Core Abstractions
    
    ### Communication Pattern Interface
    - step(): Called after each training step to perform inter-node communication
    - zero_grad(): Clears gradients before accumulation (delegates to optimizer)
    - Abstract interface allows pluggable communication strategies
    
    ### Optimizer Integration
    - Strategies wrap and manage underlying PyTorch optimizers
    - Optimizer creation deferred to _init_node() for proper device placement
    - Consistent interface across different optimization algorithms
    
    ### Learning Rate Scheduling
    - Built-in support for common LR scheduling patterns
    - Callback system for real-time LR monitoring and logging
    - Automatic integration with PyTorch scheduler ecosystem
    
    ## Design Philosophy
    
    ### Hardware Agnostic
    - Uses communicate.py abstractions for CUDA/MPS/CPU compatibility
    - Strategies don't need device-specific communication logic
    - Transparent handling of Apple Silicon MPS limitations
    
    ### Composable and Extensible
    - Clean separation between communication logic and optimization
    - Easy to implement new distributed training algorithms
    - Pluggable into existing training infrastructure
    
    ### Production Ready
    - Built-in logging and monitoring integration
    - Robust error handling and state management
    - Performance tracking with communication volume metrics
    
    ## Lifecycle Management
    
    ### Construction Phase
    1. Strategy created with optimizer specs and hyperparameters
    2. Configuration stored but no heavy objects created
    3. LR scheduler configuration prepared but not instantiated
    
    ### Node Initialization Phase
    1. _init_node() called with model, rank, and node count
    2. Optimizer created with proper device placement
    3. LR scheduler instantiated and configured
    4. Strategy-specific setup completed
    
    ### Training Phase
    1. step() called after each backward pass
    2. Communication performed according to strategy
    3. Optimizer step executed
    4. LR scheduling and callbacks triggered
    
    ## Implementation Requirements
    
    Subclasses must implement:
    - step(): Core communication and optimization logic
    - _init_node(): Strategy-specific initialization
    
    Subclasses typically override:
    - __init__(): Strategy-specific configuration
    - Additional helper methods for complex communication patterns
    
    ## Built-in Features
    
    ### Learning Rate Scheduling
    - lambda_cosine: Warmup + cosine annealing schedule
    - Custom scheduler support via lr_scheduler parameter
    - Automatic step() integration with callback system
    
    ### Monitoring Integration
    - Communication volume tracking via self.nbytes
    - LR callback system for real-time monitoring
    - Configuration extraction for experiment logging
    
    ### State Management
    - Step counting for LR scheduling
    - Max steps configuration for schedule calculation
    - Callback list management for monitoring integration
    """
    
    def __init__(
        self,
        lr_scheduler: str = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize strategy with scheduler configuration and hyperparameters.
        
        This constructor stores configuration but defers heavy object creation
        to _init_node() to ensure proper device placement and model integration.
        
        Args:
            lr_scheduler: LR scheduler type ("lambda_cosine" or scheduler class)
            lr_scheduler_kwargs: Parameters for LR scheduler configuration
            **kwargs: Additional strategy-specific hyperparameters
        """
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.kwargs = kwargs

        # Set kwargs as instance attributes for easy access
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Initialize scheduler as None; will be set after self.optim is defined in subclasses
        self.scheduler = None

        # List of callbacks to record learning rate changes
        self.lr_callbacks = []

        # Needs to be initialized for first call of lr_lambda during scheduler setup
        self.max_steps = 1

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize strategy with distributed training context.
        
        Called by TrainNode after strategy is copied to each worker process.
        This method completes the strategy initialization with model reference,
        distributed training context, and local state tracking.
        
        Args:
            model: PyTorch model being trained (already on correct device)
            rank: Process rank (0 to num_nodes-1) for distributed coordination
            num_nodes: Total number of distributed training processes
        """
        self.model = model
        self.rank = rank
        self.num_nodes = num_nodes

        # Initialize step counter for LR scheduling and communication coordination
        self.local_step = 0

    @abstractmethod
    def step(self):
        """
        Execute one step of the distributed training strategy.
        
        This is the core method that subclasses must implement to define their
        communication pattern. Called after gradients are computed but before
        they're applied to parameters.
        
        ## Standard Step Pattern
        
        All strategy implementations should follow this pattern:
        1. Reset communication metrics (self.nbytes = 0)
        2. Perform strategy-specific communication
        3. Execute optimizer step
        4. Handle LR scheduling and callbacks
        5. Update local step counter
        
        ## Communication Metrics
        
        Strategies should track communication volume in self.nbytes for monitoring:
        - Set self.nbytes = 0 at start of step
        - Add communication volume for each operation
        - Used for bandwidth monitoring and strategy comparison
        
        ## Learning Rate Management
        
        Base implementation handles:
        - Scheduler step execution (if scheduler exists)
        - LR callback notifications (rank 0 only)
        - Step counter maintenance for scheduling
        
        Subclasses must call super().step() to ensure proper LR handling.
        """
        # Reset communication volume tracking
        self.nbytes = 0

        # Apply learning rate scheduling
        if self.scheduler is not None:
            self.scheduler.step()

            # Notify LR callbacks (rank 0 only to avoid duplicate logging)
            if self.rank == 0:
                for callback in self.lr_callbacks:
                    callback(self.scheduler.get_last_lr()[0])

        # Update step counter for next iteration
        self.local_step += 1

    def zero_grad(self):
        """
        Clear gradients in preparation for backward pass.
        
        Delegates to underlying optimizer to clear gradients. Called before
        gradient accumulation begins in TrainNode._train_step().
        """
        self.optim.zero_grad()

    def _setup_scheduler(self):
        """
        Initialize learning rate scheduler with built-in cosine annealing support.
        
        This method creates LR schedulers that integrate seamlessly with distributed
        training. Supports both built-in patterns (lambda_cosine) and custom
        PyTorch scheduler classes.
        
        ## Built-in Scheduler: lambda_cosine
        
        Implements warmup + cosine annealing pattern commonly used in modern training:
        
        ### Warmup Phase (steps 0 to warmup_steps)
        - Linear scaling from 0 to base learning rate
        - Prevents early training instability
        - Default warmup_steps=1 (minimal warmup)
        
        ### Cosine Annealing Phase (warmup_steps to max_steps)
        - Smooth decay following cosine curve
        - Configurable minimum LR (default 10% of base LR)
        - Enables better convergence in final training stages
        
        ### Mathematical Formula
        ```
        if step < warmup_steps:
            lr_factor = step / warmup_steps
        elif cosine_anneal:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            cosine_term = 0.5 * (1 + cos(π * progress))
            lr_factor = (1 - min_lr_factor) * cosine_term + min_lr_factor
        else:
            lr_factor = 1.0  # constant LR after warmup
        ```
        
        ## Custom Scheduler Support
        
        Any PyTorch scheduler class can be used by passing:
        - lr_scheduler: Scheduler class (e.g., torch.optim.lr_scheduler.StepLR)
        - lr_scheduler_kwargs: Arguments for scheduler constructor
        
        ## Configuration Parameters
        
        ### lambda_cosine scheduler
        - warmup_steps: Linear warmup duration (default: 1)
        - max_steps: Total training steps for cosine decay
        - cosine_anneal: Enable cosine decay after warmup (default: False)
        - min_lr_factor: Minimum LR as fraction of base LR (fixed: 0.1)
        
        ### Custom schedulers
        - lr_scheduler_kwargs: Passed directly to scheduler constructor
        - Must be compatible with PyTorch scheduler interface
        
        ## Integration with Distributed Training
        
        ### Step Synchronization
        - All ranks call scheduler.step() simultaneously
        - Ensures consistent LR across all distributed processes
        - Critical for training stability and reproducibility
        
        ### Callback Integration
        - LR changes automatically reported to registered callbacks
        - Enables real-time monitoring in WandB, CSV logs, progress bars
        - Only rank 0 triggers callbacks to avoid duplicate logging
        
        Called by:
            Strategy subclasses during _init_node() after optimizer creation
            
        Calls:
            torch.optim.lr_scheduler classes for LR management
        """
        def lr_lambda(current_step):
            """Lambda function for cosine annealing with warmup schedule."""
            warmup_steps = self.lr_scheduler_kwargs.get("warmup_steps", 1)
            
            # Use explicit max_steps from config, otherwise use training max_steps
            if "max_steps" in self.lr_scheduler_kwargs:
                max_steps = min(self.lr_scheduler_kwargs["max_steps"], self.max_steps)
            else:
                max_steps = self.max_steps
                
            cosine_anneal = self.lr_scheduler_kwargs.get("cosine_anneal", False)

            if current_step < warmup_steps:
                # Linear warmup from 0 to base LR
                return float(current_step) / float(max(warmup_steps, 1))
            elif cosine_anneal:
                # Cosine annealing with minimum LR floor
                min_lr_factor = StrategyConstants.MIN_LR_FACTOR
                progress = (current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (1 - min_lr_factor) * cosine_term + min_lr_factor
            else:
                # Constant LR after warmup
                return 1.0

        # Create appropriate scheduler based on configuration
        if self.lr_scheduler == "lambda_cosine":
            self.scheduler = LambdaLR(self.optim, lr_lambda)
        elif self.lr_scheduler is not None:
            # Custom scheduler with user-provided arguments
            lr_sched_kwargs = (
                self.lr_scheduler_kwargs if self.lr_scheduler_kwargs is not None else {}
            )
            self.scheduler = self.lr_scheduler(self.optim, **lr_sched_kwargs)
        else:
            # No scheduler - constant learning rate
            self.scheduler = None

    def __config__(self):
        remove_keys = [
            "iteration",
            "local_step",
            "lr_callbacks",
            "model",
            "optim",
            "scheduler",
        ]

        config = super().__config__(remove_keys)

        config["strategy"] = self.__class__.__name__

        return config


class SimpleReduceStrategy(Strategy):
    """
    Basic gradient averaging strategy for traditional data parallel training.
    
    SimpleReduceStrategy implements the classic data parallel training pattern
    where gradients are averaged across all nodes after each backward pass.
    This provides the most straightforward distributed training approach with
    immediate communication after every training step.
    
    ## Algorithm Overview
    
    ### Communication Pattern
    1. **Gradient Computation**: Each node computes gradients on local data
    2. **Gradient Averaging**: all_reduce sums gradients across all nodes
    3. **Normalization**: Divide by num_nodes to get average gradients
    4. **Optimization**: Apply averaged gradients with local optimizer
    
    ### Mathematical Equivalence
    This strategy is mathematically equivalent to:
    - Training on a single machine with batch_size * num_nodes
    - Standard data parallel training in PyTorch
    - Synchronous distributed training patterns
    
    ## Key Features
    
    ### Immediate Communication
    - Gradients communicated after every backward pass
    - No communication delays or batching
    - Consistent with traditional distributed training expectations
    
    ### Gradient Clipping Integration
    - Optional gradient clipping applied after gradient averaging
    - Uses torch.nn.utils.clip_grad_norm_ for stability
    - Clipping applied to averaged gradients, not local gradients
    
    ### Optimizer Integration
    - Supports any PyTorch optimizer via OptimSpec
    - Default: AdamW optimizer for modern training practices
    - Learning rate scheduling fully supported
    
    ## Performance Characteristics
    
    ### Communication Volume
    - Highest communication overhead among strategies
    - Every step requires full gradient synchronization
    - Communication volume = gradient_size per step
    
    ### Convergence Properties
    - Identical convergence to single-machine training
    - No algorithmic approximations or delays
    - Gold standard for distributed training correctness
    
    ### Scalability
    - Communication bound at large scale
    - Works well for small to medium node counts (2-16 nodes)
    - May become bottleneck for very large distributed setups
    
    ## Implementation Details
    
    ### Gradient Handling
    - Checks param.grad is not None before communication
    - Uses in-place division for memory efficiency
    - Preserves gradient computation graph properties
    
    ### Device Compatibility
    - Uses communicate.all_reduce for hardware abstraction
    - Works with CUDA, MPS, and CPU backends
    - Automatic handling of Apple Silicon MPS limitations
    
    ### Error Resilience
    - Graceful handling of None gradients
    - Communication failures handled by underlying layers
    - Optimizer errors propagated normally
    
    ## Usage Patterns
    
    ### Basic Usage
    ```python
    strategy = SimpleReduceStrategy()  # Default AdamW optimizer
    ```
    
    ### Custom Optimizer
    ```python
    from exogym.strategy.optim import OptimSpec
    strategy = SimpleReduceStrategy(
        optim_spec=OptimSpec(torch.optim.SGD, lr=0.01, momentum=0.9)
    )
    ```
    
    ### With Gradient Clipping
    ```python
    strategy = SimpleReduceStrategy(max_norm=1.0)  # Clip gradients to norm 1.0
    ```
    
    ## Comparison with Other Strategies
    
    ### vs. DiLoCoStrategy
    - SimpleReduce: Immediate communication, higher bandwidth
    - DiLoCo: Periodic communication, lower bandwidth, similar convergence
    
    ### vs. Federated Strategies
    - SimpleReduce: Synchronous, immediate updates
    - Federated: Asynchronous, delayed updates, client-server pattern
    
    This strategy serves as both a production algorithm for moderate-scale
    distributed training and a reference implementation for testing other
    communication strategies.
    """
    
    def __init__(self, optim_spec=None, max_norm=None, **kwargs):
        """
        Initialize simple gradient averaging strategy.
        
        Args:
            optim_spec: Optimizer specification (default: AdamW)
            max_norm: Gradient clipping threshold (default: None, no clipping)
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        super().__init__(**kwargs)

        # Default to AdamW optimizer for modern training practices
        self.optim_spec = ensure_optim_spec(optim_spec) or OptimSpec(torch.optim.AdamW)

        # Optional gradient clipping for training stability
        self.max_norm = max_norm

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize strategy with model and distributed training context.
        
        Creates optimizer with proper device placement and sets up learning
        rate scheduling according to configuration.
        """
        super()._init_node(model, rank, num_nodes)

        # Create optimizer with model parameters on correct device
        self.optim = self.optim_spec.build(model)
        
        # Initialize learning rate scheduler if configured
        self._setup_scheduler()

    def step(self):
        """
        Execute gradient averaging and optimization step.
        
        Performs the core data parallel training logic:
        1. Average gradients across all nodes via all_reduce
        2. Apply gradient clipping if configured
        3. Execute optimizer step with averaged gradients
        4. Handle learning rate scheduling
        """
        # Average gradients across all nodes (including single-node case for consistency)
        if self.num_nodes > 1 or True:
            for param in self.model.parameters():
                if param.grad is not None:
                    # Sum gradients across all nodes
                    all_reduce(param.grad)
                    # Normalize to get average gradient
                    param.grad.div_(self.num_nodes)

            # Apply gradient clipping to averaged gradients if configured
            if self.max_norm:
                nn_utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )

        # Apply averaged gradients to model parameters
        self.optim.step()

        # Handle learning rate scheduling and step tracking
        super().step()
