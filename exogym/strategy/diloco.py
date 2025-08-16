"""
ExoGym DiLoCoStrategy - Distributed Low-Communication Training Implementation

This module implements the DiLoCo (Distributed Low-Communication) training strategy,
which reduces communication overhead in distributed training by using a two-level
optimization approach with periodic model averaging.

## DiLoCo Algorithm Overview

DiLoCo reduces distributed training communication by orders of magnitude compared
to traditional gradient averaging approaches. Instead of communicating after every
step, it uses:

1. **Inner Optimization**: Local SGD/Adam updates on each node for H steps
2. **Outer Optimization**: Periodic model averaging and outer optimizer update
3. **Communication Schedule**: Full model communication only every H steps

## Key Innovation: Two-Level Optimization

### Inner Optimizer (per-node)
- Runs standard optimization (AdamW, SGD, etc.) on local gradients
- No communication during inner steps - pure local training
- Accumulates H steps of local model evolution

### Outer Optimizer (global)  
- Operates on model differences after H inner steps
- Uses momentum-based updates (SGD with Nesterov momentum)
- Only rank 0 performs outer optimization, then broadcasts result

## Mathematical Framework

At communication step (every H inner steps):
1. **Average Models**: θ_avg = (1/N) * Σ θ_i across all nodes
2. **Compute Pseudo-Gradients**: g_outer = θ_master - θ_avg  
3. **Outer Step**: θ_master = OuterOptimizer.step(θ_master, g_outer)
4. **Broadcast**: θ_i = θ_master for all nodes

## Architecture Details

### Master Model (Rank 0 Only)
- CPU-resident copy of the model maintained only on rank 0
- Used to compute pseudo-gradients and apply outer optimizer updates
- Synchronized back to all nodes after outer optimization

### Rank Responsibilities
- **Rank 0**: Maintains master model, performs outer optimization, broadcasts results
- **All Ranks**: Perform inner optimization, participate in model averaging

### Communication Pattern
```
Steps 1-H: Local training only (no communication)
Step H: all_reduce(models) → outer_optimization() → broadcast(result)
Steps H+1-2H: Local training only...
```

## Hyperparameter: H (Communication Interval)

The H parameter controls the communication frequency:
- **Small H** (1-10): More communication, closer to traditional data parallel
- **Medium H** (50-200): Balanced communication reduction with convergence  
- **Large H** (500+): Minimal communication, may hurt convergence

Typical values: H=100 provides good communication reduction while maintaining
convergence properties similar to data parallel training.

## Implementation Details

### Device Management
- Master model kept on CPU to avoid GPU memory overhead
- Automatic data movement between GPU (training) and CPU (master model)
- Handles CUDA, MPS, and CPU backends transparently

### Gradient Clipping Integration
- Applied before inner optimizer step, not before outer step
- Uses standard torch.nn.utils.clip_grad_norm_ on local gradients

### Scheduler Integration  
- LR scheduling applied to inner optimizer as normal
- Outer optimizer typically uses fixed LR with momentum

## Called by:
- TrainNode.step() during distributed training execution
- Coordinates with other DiLoCoStrategy instances across nodes

## Calls:
- communicate.all_reduce() for model averaging every H steps
- communicate.broadcast() for distributing outer optimizer results
- Inner/outer optimizers for parameter updates

## Performance Characteristics

### Communication Volume
- Reduces communication by factor of H compared to gradient averaging
- Each communication sends full model parameters (not gradients)
- Total bandwidth: (model_size * steps) / H

### Memory Overhead
- Rank 0: Additional CPU copy of model (~2x model memory)
- Other ranks: Standard training memory footprint
- No significant memory scaling with H parameter

### Convergence Properties
- Maintains similar convergence to data parallel training for appropriate H
- May require learning rate scaling: outer_lr = √(H) * base_lr
- Works best with momentum-based outer optimizers

## Research Reference

Based on "Distributed Low-Communication Training" research, this implementation
provides a production-ready version of the DiLoCo algorithm with hardware-agnostic
distributed communication and integration with modern PyTorch training pipelines.
"""

import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from typing import Optional, Union

from .strategy import Strategy
from .optim import OptimSpec, ensure_optim_spec
from .communicate import all_reduce, broadcast


class DiLoCoStrategy(Strategy):
    """
    Distributed Low-Communication (DiLoCo) training strategy implementation.
    
    DiLoCo dramatically reduces communication overhead in distributed training by using
    a two-level optimization approach with periodic model averaging. Instead of
    communicating gradients after every step, nodes train independently for H steps
    then synchronize model parameters.
    
    ## Core Algorithm
    
    ### Two-Level Optimization Structure
    - **Inner Optimizer**: Local optimization (AdamW, SGD, etc.) on each node
    - **Outer Optimizer**: Global optimization on averaged model updates
    - **Communication Schedule**: Full model sync every H steps only
    
    ### Mathematical Framework
    
    For communication rounds at steps H, 2H, 3H, etc.:
    1. **Model Averaging**: θ̄ = (1/N) * Σᵢ θᵢ (average across all nodes)
    2. **Pseudo-Gradient**: g = θ_master - θ̄ (difference from average)
    3. **Outer Update**: θ_master ← OuterOptim(θ_master, g)
    4. **Broadcast**: θᵢ ← θ_master for all nodes i
    
    Between communication rounds, each node performs standard local training.
    
    ## Key Innovation: Communication Reduction
    
    ### Traditional Data Parallel
    - Communication: Every step (gradient averaging)
    - Volume: gradient_size * steps
    - Bandwidth: Constant high usage
    
    ### DiLoCo
    - Communication: Every H steps (model averaging)
    - Volume: model_size * (steps/H)
    - Bandwidth: Reduced by factor of H
    
    For H=100, this represents a 100x reduction in communication frequency.
    
    ## Architecture Details
    
    ### Master Model (Rank 0 Only)
    - Maintained on CPU to avoid GPU memory overhead
    - Receives averaged model from all nodes
    - Applies outer optimizer updates
    - Broadcasts result back to all nodes
    
    ### Rank-Based Responsibilities
    - **Rank 0**: Manages master model, performs outer optimization, broadcasts
    - **All Ranks**: Perform inner optimization, participate in averaging
    
    ### Device Management
    - Training models remain on original devices (CUDA/MPS/CPU)
    - Master model kept on CPU for memory efficiency
    - Automatic data movement handled transparently
    
    ## Hyperparameter: H (Communication Interval)
    
    ### Choosing H Value
    - **H=1**: Equivalent to standard data parallel (no communication reduction)
    - **H=10-50**: Moderate communication reduction, minimal convergence impact
    - **H=100-200**: Significant reduction, good convergence for most models
    - **H=500+**: Maximum reduction, may impact convergence quality
    
    ### H Selection Guidelines
    - Larger models: Can tolerate larger H (more local computation capacity)
    - Complex datasets: May need smaller H (more frequent synchronization)
    - Network constraints: Larger H for bandwidth-limited environments
    - Convergence sensitivity: Start with H=100, adjust based on validation metrics
    
    ## Convergence Properties
    
    ### Theoretical Guarantees
    - Maintains convergence rates similar to centralized training
    - Convergence depends on H, learning rates, and problem characteristics
    - Proven effective for convex and non-convex optimization
    
    ### Practical Performance
    - Works well for transformer language models
    - Effective for computer vision tasks
    - May require outer LR tuning: typically √H scaling from base LR
    
    ## Implementation Optimizations
    
    ### Memory Efficiency
    - Master model only on rank 0 (not replicated across all ranks)
    - CPU storage prevents GPU memory pressure
    - Temporary tensors created only during communication
    
    ### Communication Optimization
    - Model parameters communicated, not gradients
    - Single all_reduce followed by broadcast pattern
    - Leverages high-bandwidth interconnects efficiently
    
    ### Gradient Clipping Integration
    - Applied to inner optimizer gradients before local steps
    - Outer optimizer uses model differences, not traditional gradients
    - Maintains training stability without interfering with outer optimization
    
    ## Usage Patterns
    
    ### Basic Usage
    ```python
    strategy = DiLoCoStrategy(H=100)  # Default AdamW inner, SGD outer
    ```
    
    ### Custom Optimizers
    ```python
    strategy = DiLoCoStrategy(
        optim_spec=OptimSpec(torch.optim.SGD, lr=0.01),
        outer_optim_spec=OptimSpec(torch.optim.AdamW, lr=0.1),
        H=200
    )
    ```
    
    ### Learning Rate Scaling
    ```python
    # Common pattern: scale outer LR by sqrt(H)
    base_lr = 0.001
    H = 100
    strategy = DiLoCoStrategy(
        optim_spec=OptimSpec(torch.optim.AdamW, lr=base_lr),
        outer_optim_spec=OptimSpec(torch.optim.SGD, lr=base_lr * math.sqrt(H)),
        H=H
    )
    ```
    
    ## Performance Characteristics
    
    ### Communication Volume
    - Reduces total communication by factor of H
    - Each communication sends full model (not gradients)
    - Total bandwidth: model_size * training_steps / H
    
    ### Computational Overhead
    - Negligible overhead during local training steps
    - Model averaging overhead every H steps
    - Outer optimization minimal (SGD typically used)
    
    ### Memory Usage
    - Rank 0: Additional model copy on CPU (~2x model memory)
    - Other ranks: Standard training memory usage
    - No scaling with H parameter
    
    This implementation provides a production-ready version of the DiLoCo algorithm
    with full integration into the ExoGym distributed training framework.
    """
    
    def __init__(
        self,
        optim_spec: Optional[Union[str, OptimSpec]] = None, # inner optimizer is named optim_spec for consistency
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        H: int = 100,
        **kwargs,
    ):
        """
        Initialize DiLoCo strategy with inner/outer optimizers and communication interval.
        
        Args:
            optim_spec: Inner optimizer specification (default: AdamW)
            outer_optim_spec: Outer optimizer specification (default: SGD with Nesterov)
            H: Communication interval in training steps (default: 100)
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        # Default inner optimizer: AdamW for modern training practices
        self.inner_optim_spec = ensure_optim_spec(
            optim_spec, OptimSpec(torch.optim.AdamW)
        )
        
        # Default outer optimizer: SGD with Nesterov momentum at higher LR
        # LR chosen based on research recommendations for DiLoCo
        from .strategy import StrategyConstants
        self.outer_optim_spec = ensure_optim_spec(
            outer_optim_spec, OptimSpec(torch.optim.SGD, lr=StrategyConstants.DEFAULT_OUTER_LR, nesterov=True, momentum=StrategyConstants.DEFAULT_SGD_MOMENTUM)
        )

        # Communication interval: balance between communication reduction and convergence
        self.H = H

        super().__init__(**kwargs)

    def _average_models(self) -> None:
        """
        Average model parameters across all nodes via all_reduce operation.
        
        This method implements the first step of DiLoCo's outer optimization:
        computing the average model θ̄ = (1/N) * Σᵢ θᵢ across all nodes.
        
        The averaging is performed in-place on self.model parameters, converting
        each node's local model to the global average. This averaged model is then
        used by rank 0 to compute pseudo-gradients for the outer optimizer.
        
        ## Implementation Details
        
        - Uses all_reduce with SUM operation for efficiency
        - Divides by num_nodes to get true average (not sum)
        - Operates directly on param.data to avoid gradient computation
        - Preserves parameter shapes and device placement
        
        Called by:
            DiLoCoStrategy.step() every H steps during communication rounds
            
        Calls:
            communicate.all_reduce() for distributed parameter averaging
        """
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.num_nodes

    def _broadcast_model_params(self) -> None:
        """
        Broadcast updated model parameters from rank 0 to all other nodes.
        
        This method implements the final step of DiLoCo's outer optimization:
        distributing the updated master model from rank 0 to all other nodes.
        After the outer optimizer updates the master model, all nodes must
        synchronize to this new model state.
        
        ## Implementation Details
        
        - Uses broadcast operation with rank 0 as source
        - Overwrites local model parameters with master model values
        - Ensures all nodes resume training from identical model state
        - Maintains parameter device placement on each node
        
        Called by:
            DiLoCoStrategy.step() every H steps after outer optimization
            
        Calls:
            communicate.broadcast() for parameter distribution
        """
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        """
        Compute pseudo-gradients for outer optimizer from model differences.
        
        This method implements the core innovation of DiLoCo: treating the difference
        between the master model and averaged model as pseudo-gradients for the
        outer optimizer. The pseudo-gradient g = θ_master - θ̄ represents the
        direction the outer optimizer should move.
        
        ## Mathematical Background
        
        Traditional gradients: g = ∂L/∂θ (derivative of loss w.r.t. parameters)
        DiLoCo pseudo-gradients: g = θ_master - θ̄ (difference from average)
        
        These pseudo-gradients capture the "disagreement" between the master model
        and the averaged node models, providing a signal for outer optimization.
        
        ## Implementation Details
        
        - Operates on master_model.named_parameters() to set .grad attributes
        - Computes difference: master_param - averaged_param
        - Transfers averaged parameters from GPU to CPU for comparison
        - Creates proper gradient tensors for outer optimizer
        
        Called by:
            DiLoCoStrategy.step() every H steps before outer optimizer step
            
        Requires:
            _average_models() must be called first to compute θ̄
        """
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data.to("cpu")

    def _synchronize_master_model(self) -> None:
        """
        Copy updated master model parameters back to training model.
        
        After the outer optimizer updates the master model on rank 0, this method
        copies the updated parameters back to the training model. This ensures
        that rank 0's training model matches the updated master model before
        broadcasting to other nodes.
        
        ## Implementation Details
        
        - Copies from CPU master model to GPU training model
        - Handles automatic device placement (CPU → GPU/MPS/CPU)
        - Updates self.model in-place to preserve object references
        - Maintains parameter metadata and gradient computation graph
        
        Called by:
            DiLoCoStrategy.step() every H steps after outer optimizer update
            
        Calls:
            Automatic device conversion via .to(param.device)
        """
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)

    def step(self):
        """
        Execute DiLoCo training step with inner optimization and periodic outer communication.
        
        This method implements the complete DiLoCo algorithm, coordinating inner optimization
        (every step) with outer optimization (every H steps). The two-level structure
        enables dramatic communication reduction while maintaining convergence properties.
        
        ## Step Execution Flow
        
        ### Every Step: Inner Optimization
        1. **Gradient Clipping**: Apply clipping to local gradients if configured
        2. **Inner Step**: Update model parameters using inner optimizer (AdamW, SGD, etc.)
        
        ### Every H Steps: Outer Communication
        1. **Model Averaging**: Compute global average θ̄ across all nodes
        2. **Outer Optimization (Rank 0 only)**:
           - Compute pseudo-gradients: g = θ_master - θ̄
           - Apply outer optimizer step to master model
           - Synchronize training model with updated master model
        3. **Broadcasting**: Distribute updated model from rank 0 to all nodes
        
        ## Communication Schedule
        
        ### Local Training Phase (steps 1 to H-1, H+1 to 2H-1, etc.)
        - Pure local training with no inter-node communication
        - Each node evolves independently using inner optimizer
        - Maximum communication bandwidth savings
        
        ### Communication Phase (steps H, 2H, 3H, etc.)
        - All nodes participate in model averaging
        - Rank 0 performs outer optimization
        - All nodes synchronize to updated model
        - Brief communication burst followed by local training
        
        ## Rank-Specific Responsibilities
        
        ### All Ranks
        - Perform inner optimization every step
        - Participate in model averaging every H steps
        - Receive updated model via broadcast
        
        ### Rank 0 (Master Node)
        - Maintains master model on CPU for outer optimization
        - Computes pseudo-gradients from model differences
        - Applies outer optimizer updates
        - Sources broadcast of updated model
        
        ## Gradient Clipping Integration
        
        Gradient clipping (if enabled) is applied to inner optimizer gradients:
        - Applied before inner optimizer step
        - Uses standard torch.nn.utils.clip_grad_norm_
        - Does not interfere with outer optimizer pseudo-gradients
        - Maintains training stability during local training phases
        
        ## Performance Characteristics
        
        ### Communication Efficiency
        - Communication occurs only every H steps
        - Reduces bandwidth usage by factor of H
        - Single all_reduce + broadcast pattern per communication
        
        ### Computational Overhead
        - Inner steps: identical to standard training
        - Outer steps: minimal overhead (model averaging + SGD step)
        - Total overhead: negligible compared to training computation
        
        ## Error Handling
        
        ### Communication Failures
        - all_reduce failures affect all nodes simultaneously
        - broadcast failures prevent model synchronization
        - Robust error handling delegated to communicate.py layer
        
        ### Optimization Failures
        - Inner optimizer failures affect individual nodes
        - Outer optimizer failures (rank 0) affect global training
        - Standard PyTorch optimizer error handling applies
        
        Called by:
            TrainNode._train_step() after gradient computation
            
        Calls:
            Inner optimizer for local parameter updates
            Outer optimizer for global model updates (rank 0 only)
            communicate.py functions for distributed operations
        """
        # Apply gradient clipping to local gradients if configured
        if "max_norm" in self.kwargs:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.kwargs["max_norm"]
            )

        # Inner optimization step: apply local gradients to model parameters
        # This happens every step regardless of communication schedule
        self.optim.step()

        # Outer optimization step: periodic model averaging and global update
        # Only occurs every H steps starting from step H (not step 0)
        if self.local_step % self.H == 0 and self.local_step > 0:
            # Step 1: Average models across all nodes
            # Converts each node's local model to global average
            self._average_models()

            # Step 2: Outer optimization (rank 0 only)
            if self.rank == 0:
                # Prepare outer optimizer for pseudo-gradient computation
                self.outer_optimizer.zero_grad()
                
                # Compute pseudo-gradients from model differences
                self._set_master_grad()
                
                # Apply outer optimizer update to master model
                self.outer_optimizer.step()
                
                # Copy updated master model back to training model
                self._synchronize_master_model()

            # Step 3: Broadcast updated model from rank 0 to all nodes
            # Ensures all nodes resume training from identical model state
            self._broadcast_model_params()

        # Handle learning rate scheduling and step tracking
        super().step()

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize DiLoCo strategy with model and distributed training context.
        
        This method completes the DiLoCo setup by creating optimizers and, for rank 0,
        initializing the master model that handles outer optimization. The master model
        is kept on CPU to minimize GPU memory usage.
        
        ## Initialization Steps
        
        ### Rank 0 (Master Node) Setup
        1. **Master Model Creation**: Deep copy training model to CPU
        2. **Gradient Enablement**: Ensure master model parameters can receive gradients
        3. **Outer Optimizer**: Create optimizer for master model updates
        
        ### All Ranks Setup
        1. **Inner Optimizer**: Create optimizer for local training model
        2. **LR Scheduler**: Initialize learning rate scheduling (inner optimizer only)
        
        ## Master Model Design (Rank 0 Only)
        
        ### CPU Placement Strategy
        - Avoids GPU memory overhead (important for large models)
        - Enables efficient outer optimization without device conflicts
        - Automatic CPU↔GPU transfers handled transparently
        
        ### Gradient Configuration
        - requires_grad=True ensures parameters can receive pseudo-gradients
        - Enables standard PyTorch optimizer usage for outer optimization
        - Maintains gradient computation graph for outer optimizer
        
        ## Memory Footprint
        
        ### Rank 0
        - Training model: Standard GPU memory usage
        - Master model: Additional CPU memory (~equal to training model size)
        - Total: ~2x model memory (1x GPU + 1x CPU)
        
        ### Other Ranks
        - Training model: Standard GPU memory usage
        - No master model overhead
        - Total: 1x model memory
        
        ## Optimizer Configuration
        
        ### Inner Optimizer (All Ranks)
        - Built from inner_optim_spec (default: AdamW)
        - Operates on GPU/MPS training model
        - Handles local parameter updates every step
        
        ### Outer Optimizer (Rank 0 Only)
        - Built from outer_optim_spec (default: SGD with Nesterov)
        - Operates on CPU master model
        - Handles global parameter updates every H steps
        
        Called by:
            Strategy base class after strategy is copied to worker process
            
        Calls:
            OptimSpec.build() for optimizer creation
            Strategy._setup_scheduler() for LR scheduling
        """
        super()._init_node(model, rank, num_nodes)

        # Initialize master model and outer optimizer on rank 0 only
        if self.rank == 0:
            # Create CPU copy of model for outer optimization
            self.master_model = deepcopy(model).to("cpu")
            
            # Enable gradients for outer optimizer usage
            for param in self.master_model.parameters():
                param.requires_grad = True

            # Create outer optimizer for master model updates
            self.outer_optimizer = self.outer_optim_spec.build(self.master_model)

        # Create inner optimizer for all ranks (local training)
        self.optim = self.inner_optim_spec.build(model)
        
        # Initialize learning rate scheduler for inner optimizer
        self._setup_scheduler()
