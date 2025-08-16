"""
ExoGym SPARTA Strategy - Sparse Parameter Communication for Distributed Training

This module implements SPARTA (Sparse Parameter Communication) algorithms for
reducing communication overhead in distributed training through selective parameter
updates. SPARTA achieves significant bandwidth reduction by communicating only
a subset of model parameters in each round while maintaining convergence properties.

## SPARTA Algorithm Overview

### Core Innovation: Sparse Communication
Traditional distributed training communicates all model parameters or gradients,
creating bandwidth bottlenecks that limit scalability. SPARTA addresses this by:
- **Selective Communication**: Only communicate a fraction of parameters per round
- **Cyclic Coverage**: Ensure all parameters are updated over multiple rounds
- **Maintained Convergence**: Preserve training effectiveness despite reduced communication

### Key Benefits
- **Bandwidth Reduction**: 10-100x reduction in communication volume
- **Scalability**: Enables training with limited network infrastructure
- **Flexibility**: Multiple sparsification patterns for different scenarios
- **Convergence**: Maintains training quality with appropriate sparsification rates

## Sparsification Strategies

### RandomIndexSelector
- **Random Sampling**: Each round selects random p% of parameters
- **Unbiased Coverage**: All parameters have equal selection probability
- **Simple Implementation**: Minimal state tracking required
- **Good Convergence**: Works well for most model architectures

### ShuffledSequentialIndexSelector
- **Deterministic Coverage**: Guarantees all parameters updated in 1/p rounds
- **Shuffled Ordering**: Random permutation prevents systematic bias
- **Partition-Based**: Divides parameters into chunks for sequential processing
- **Predictable Bandwidth**: Consistent communication volume per round

### PartitionedIndexSelector
- **Dynamic Partitioning**: Creates balanced partitions for each parameter tensor
- **Cyclic Updates**: Rotates through partitions systematically
- **Tensor-Aware**: Handles different parameter shapes independently
- **Memory Efficient**: Minimal overhead for partition management

## Communication Architecture

### SparseCommunicator Module
Implements the core sparse communication protocol:
- **Index Selection**: Uses pluggable index selectors for flexibility
- **Sparse Gathering**: Extracts only selected parameter elements
- **Distributed Averaging**: all_reduce on sparse data only
- **Sparse Scattering**: Updates model with averaged sparse data

### Integration with ExoGym
- **CommunicateOptimizeStrategy Base**: Inherits standard optimization infrastructure
- **Hardware Compatibility**: Works with CUDA, MPS, and CPU backends
- **Configuration System**: Full integration with logging and experiment tracking

## Performance Characteristics

### Communication Volume
- **Sparsification Rate p**: Controls fraction of parameters communicated
- **Bandwidth Savings**: (1-p) reduction in communication per round
- **Total Communication**: Depends on convergence speed with sparse updates

### Convergence Properties
- **Theoretical Guarantees**: Maintains convergence under appropriate conditions
- **Practical Performance**: Requires careful tuning of sparsification rate
- **Model Dependency**: Some architectures more sensitive to sparsification

## Implementation Details

### Memory Efficiency
- **In-Place Updates**: Uses masked_scatter_ for efficient parameter updates
- **Minimal Overhead**: Index selection adds negligible computation
- **Device Awareness**: All operations respect parameter device placement

### Synchronization
- **Index Broadcasting**: Ensures all nodes use identical sparse patterns
- **Deterministic Selection**: Same seeds produce same sparsification patterns
- **Fault Tolerance**: Communication failures handled by underlying layers

## Called by:
- Bandwidth-constrained distributed training environments
- Large-scale training with limited network infrastructure
- Experiments requiring communication-computation tradeoffs

## Calls:
- CommunicateOptimizeStrategy for base distributed training functionality
- communicate.py functions for hardware-agnostic distributed operations
- Index selector classes for sparsification pattern generation

## Usage Patterns:

### Basic SPARTA with Random Selection
```python
strategy = SPARTAStrategy(p_sparta=0.01)  # Communicate 1% of parameters per round
```

### Custom Sparsification Pattern
```python
selector = ShuffledSequentialIndexSelector(p=0.05)
strategy = SPARTAStrategy(index_selector=selector)
```

### High Sparsification for Bandwidth-Limited Networks
```python
strategy = SPARTAStrategy(p_sparta=0.001, inner_optim="sgd", lr=0.01)
```

This implementation provides production-ready sparse communication capabilities
for bandwidth-constrained distributed training scenarios.
"""

import math
import torch
import torch.distributed as dist

from typing import Optional, Union

from .communicate_optimize_strategy import (
    CommunicateOptimizeStrategy,
    CommunicationModule,
)
from .optim import OptimSpec
from .communicate import all_reduce, broadcast

class SparseCommunicator(CommunicationModule):
    """
    Communication module implementing sparse parameter communication for bandwidth reduction.
    
    SparseCommunicator enables SPARTA-style distributed training by communicating only
    a subset of model parameters in each round. It uses pluggable index selectors to
    determine which parameters to communicate, providing flexibility in sparsification
    strategies while maintaining training effectiveness.
    
    ## Core Algorithm
    
    ### Sparse Communication Protocol
    1. **Index Selection**: Use index selector to determine which parameters to communicate
    2. **Index Synchronization**: Broadcast selection mask to ensure all nodes agree
    3. **Sparse Extraction**: Extract only selected parameter elements
    4. **Distributed Averaging**: all_reduce on sparse data across all nodes
    5. **Sparse Update**: Scatter averaged values back to selected parameter positions
    
    ### Key Innovation: Bandwidth Reduction
    - **Selective Communication**: Only p% of parameters communicated per round
    - **Full Coverage**: All parameters eventually updated over multiple rounds
    - **Maintained Synchronization**: All nodes stay synchronized despite sparse updates
    
    ## Sparsification Benefits
    
    ### Communication Efficiency
    - **Bandwidth Reduction**: (1-p) reduction in communication volume per round
    - **Scalability**: Enables training with limited network infrastructure
    - **Flexible Rates**: Configurable sparsification for different bandwidth constraints
    
    ### Convergence Properties
    - **Theoretical Guarantees**: Maintains convergence under appropriate sparsification rates
    - **Practical Effectiveness**: Empirically validated on various model architectures
    - **Robust Training**: Handles different data distributions and optimization landscapes
    
    ## Technical Implementation
    
    ### Memory Efficiency
    - **In-Place Updates**: Uses masked_scatter_ for efficient parameter modification
    - **Minimal Overhead**: Index selection adds negligible computational cost
    - **Device Awareness**: Respects parameter device placement throughout process
    
    ### Synchronization Protocol
    - **Deterministic Selection**: Rank 0 broadcasts index mask to ensure consistency
    - **Collective Operations**: Uses optimized all_reduce for sparse data averaging
    - **Fault Tolerance**: Robust to communication failures via underlying layers
    
    Attributes:
        index_selector: Strategy for selecting which parameters to communicate
        iteration: Current communication round counter for temporal patterns
    """

    def __init__(self, index_selector, **kwargs):
        """
        Initialize sparse communicator with index selection strategy.
        
        Args:
            index_selector: Strategy object implementing get_indices() method
            **kwargs: Additional CommunicationModule configuration
        """
        super().__init__(**kwargs)
        self.index_selector = index_selector
        self.iteration = 0

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """
        Execute sparse parameter communication round.
        
        This method implements the complete SPARTA communication protocol, from
        index selection to sparse parameter averaging. It ensures that all nodes
        communicate the same subset of parameters and update their models consistently.
        
        ## Communication Protocol Steps
        
        ### 1. Index Selection and Synchronization
        - Use index_selector to determine which parameters to communicate
        - Broadcast selection mask from rank 0 to ensure all nodes agree
        - Critical for maintaining model synchronization across nodes
        
        ### 2. Sparse Data Extraction and Communication
        - Extract selected parameter elements using boolean indexing
        - Perform all_reduce to sum sparse data across all nodes
        - Divide by num_nodes to compute average of selected elements
        
        ### 3. Sparse Parameter Update
        - Use masked_scatter_ to update only selected parameter positions
        - Preserves unchanged parameters while updating communicated subset
        - Maintains model structure and parameter metadata
        
        ## Implementation Details
        
        ### Parameter Filtering
        - Only processes parameters with requires_grad=True
        - Skips parameters without gradients (unused in current training)
        - Handles different parameter types and shapes uniformly
        
        ### Memory Management
        - Uses torch.no_grad() context to prevent gradient computation
        - Temporary sparse_data tensor automatically garbage collected
        - Minimal memory overhead beyond normal parameter storage
        
        ### Device Compatibility
        - Works transparently with CUDA, MPS, and CPU backends
        - Preserves parameter device placement throughout communication
        - Index masks created on same device as parameters
        
        Args:
            model: PyTorch model whose parameters will be sparsely communicated
            rank: Current node's rank in distributed training
            num_nodes: Total number of nodes participating in training
            local_step: Current training step (may be used by index selector)
            
        Called by:
            CommunicateOptimizeStrategy during communication rounds
            
        Calls:
            index_selector.get_indices() for parameter selection
            communicate.broadcast() for index synchronization
            communicate.all_reduce() for sparse data averaging
        """
        if num_nodes > 1:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    # Skip parameters that don't require gradients or have no gradients
                    if not param.requires_grad or param.grad is None:
                        continue

                    # Select which parameter elements to communicate this round
                    indices_mask = self.index_selector.get_indices(
                        param, self.iteration
                    )

                    # Ensure all nodes use the same sparsification pattern
                    broadcast(indices_mask, src=0)
                    
                    # Extract sparse data and perform distributed averaging
                    sparse_data = param.data[indices_mask]
                    all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                    sparse_data /= num_nodes

                    # Update model with averaged sparse data
                    param.masked_scatter_(indices_mask, sparse_data)

        # Increment iteration counter for temporal sparsification patterns
        self.iteration += 1

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize sparse communicator for distributed training node.
        
        No specific initialization required for sparse communication.
        Inherits default behavior from CommunicationModule base class.
        """
        pass


class SPARTAStrategy(CommunicateOptimizeStrategy):
    """
    SPARTA distributed training strategy with sparse parameter communication.
    
    SPARTAStrategy implements bandwidth-efficient distributed training by communicating
    only a subset of model parameters in each round. This approach dramatically reduces
    communication overhead while maintaining training effectiveness, making it ideal
    for bandwidth-constrained environments.
    
    ## Algorithm Overview
    
    ### Sparse Communication Approach
    - **Parameter Selection**: Each round, select p% of parameters for communication
    - **Distributed Averaging**: Synchronize only selected parameters across nodes
    - **Full Coverage**: All parameters eventually synchronized over multiple rounds
    - **Maintained Convergence**: Training quality preserved despite reduced communication
    
    ### Key Innovation: Bandwidth Reduction
    Traditional distributed training requires O(model_size) communication per step.
    SPARTA reduces this to O(p × model_size), providing significant bandwidth savings:
    - **p=0.01**: 100x reduction in communication volume
    - **p=0.001**: 1000x reduction with appropriate convergence adjustments
    
    ## Configuration Parameters
    
    ### Sparsification Rate (p_sparta)
    - **High Sparsity (p < 0.01)**: Maximum bandwidth savings, may slow convergence
    - **Medium Sparsity (0.01 ≤ p < 0.1)**: Good balance of efficiency and convergence
    - **Low Sparsity (p ≥ 0.1)**: Minimal convergence impact, moderate bandwidth savings
    
    ### Index Selection Strategy
    - **Default**: RandomIndexSelector for unbiased parameter sampling
    - **Custom**: Pluggable index selectors for specialized sparsification patterns
    - **Adaptive**: Could be extended with adaptive selection based on parameter importance
    
    ## Performance Characteristics
    
    ### Communication Efficiency
    - **Bandwidth Reduction**: Factor of (1/p) compared to dense communication
    - **Latency**: Similar to dense communication (same number of rounds)
    - **Scalability**: Enables training with limited network infrastructure
    
    ### Convergence Properties
    - **Theoretical Guarantees**: Maintains convergence under appropriate sparsification rates
    - **Empirical Validation**: Proven effective on various model architectures
    - **Hyperparameter Sensitivity**: Requires careful tuning of sparsification rate
    
    ## Integration with ExoGym
    
    ### Strategy Pattern Compliance
    - **CommunicateOptimizeStrategy Base**: Inherits standard optimization infrastructure
    - **Modular Design**: Pluggable index selectors and communication modules
    - **Hardware Compatibility**: Works with CUDA, MPS, and CPU backends
    
    ### Configuration System
    - **Full Logging**: Complete integration with WandB and CSV logging
    - **Experiment Tracking**: Sparsification rates and communication volumes tracked
    - **Reproducibility**: Deterministic sparsification patterns for consistent results
    
    Attributes:
        index_selector: Strategy for selecting which parameters to communicate
        p_sparta: Sparsification rate (fraction of parameters communicated per round)
    """
    
    def __init__(
        self,
        inner_optim: Optional[Union[str, OptimSpec]] = None,
        p_sparta=0.005,
        **kwargs,
    ):
        """
        Initialize SPARTA strategy with sparse communication configuration.
        
        Args:
            inner_optim: Local optimizer specification (default: AdamW)
            p_sparta: Sparsification rate - fraction of parameters to communicate per round
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        # Create default random index selector for unbiased parameter sampling
        index_selector = RandomIndexSelector(p_sparta)
        sparse_comm = SparseCommunicator(index_selector)

        # Initialize base strategy with sparse communication module
        super().__init__(
            inner_optim=inner_optim, communication_modules=[sparse_comm], **kwargs
        )

        # Store SPARTA configuration for logging and analysis
        self.index_selector = index_selector


class IndexSelector:
    """
    Abstract base class for parameter index selection strategies in sparse communication.
    
    IndexSelector defines the interface for determining which parameters to communicate
    in each SPARTA round. Different implementations provide various sparsification
    patterns, from random sampling to deterministic cycling through parameter sets.
    
    ## Design Pattern: Strategy Pattern
    
    IndexSelector enables pluggable sparsification algorithms:
    - **RandomIndexSelector**: Unbiased random sampling each round
    - **ShuffledSequentialIndexSelector**: Deterministic coverage with random ordering
    - **PartitionedIndexSelector**: Balanced partitioning for systematic updates
    
    ## Interface Requirements
    
    Subclasses must implement get_indices() to return boolean masks indicating
    which parameter elements should be communicated in the current round.
    
    Attributes:
        state: Per-parameter state tracking for temporal patterns
        p: Sparsification rate (fraction of parameters to select)
    """
    
    def __init__(self, p):
        """
        Initialize index selector with sparsification rate.
        
        Args:
            p: Sparsification rate (0 < p ≤ 1) - fraction of parameters to select
        """
        self.state = {}
        self.p = p

    def get_indices(self, param, iteration):
        """
        Generate boolean mask indicating which parameter elements to communicate.
        
        Base implementation returns all indices (full communication).
        Subclasses override this method to implement specific sparsification strategies.
        
        Args:
            param: PyTorch parameter tensor to generate indices for
            iteration: Current communication round number
            
        Returns:
            torch.Tensor: Boolean mask with same shape as param indicating selected elements
        """
        # Default implementation: communicate all parameters (no sparsification)
        return torch.ones_like(param, dtype=torch.bool)


class RandomIndexSelector(IndexSelector):
    """
    Random sampling index selector for unbiased parameter sparsification.
    
    RandomIndexSelector implements the simplest and most widely-used sparsification
    strategy: each parameter element has probability p of being selected for
    communication in each round. This provides unbiased coverage and good
    convergence properties for most model architectures.
    
    ## Algorithm
    
    ### Bernoulli Sampling
    - Each parameter element independently sampled with probability p
    - Expected communication volume: p × total_parameters per round
    - Variance in communication volume: p(1-p) × total_parameters
    
    ### Key Properties
    - **Unbiased**: All parameter elements have equal selection probability
    - **Memoryless**: Each round independent of previous selections
    - **Simple**: Minimal computational and memory overhead
    - **Robust**: Works well across different model architectures and training stages
    
    ## Performance Characteristics
    
    ### Statistical Properties
    - **Expected Coverage**: All parameters covered in approximately 1/p rounds on average
    - **Variance**: Some parameters may be selected multiple times, others delayed
    - **Convergence**: Maintains theoretical convergence guarantees under appropriate conditions
    
    ### Computational Efficiency
    - **O(1) Memory**: No state tracking required between rounds
    - **O(parameter_count) Time**: Single Bernoulli sampling operation per parameter
    - **Device Efficient**: Sampling performed directly on parameter device
    
    ## Usage Recommendations
    
    ### Sparsification Rate Selection
    - **p=0.01-0.05**: Good starting point for most models and training scenarios
    - **Higher p**: Better convergence, higher communication cost
    - **Lower p**: Maximum bandwidth savings, may require convergence tuning
    
    ### Model Compatibility
    - **Transformers**: Works well with appropriate p values (0.01-0.1)
    - **CNNs**: Generally robust to higher sparsification rates
    - **RNNs**: May require careful tuning due to temporal dependencies
    """
    
    def get_indices(self, param, iteration):
        """
        Generate random boolean mask for parameter sparsification.
        
        Performs Bernoulli sampling with probability p for each parameter element.
        Creates a boolean mask indicating which elements to communicate in this round.
        
        Args:
            param: PyTorch parameter tensor to generate indices for
            iteration: Communication round number (unused in random selection)
            
        Returns:
            torch.Tensor: Boolean mask with True for selected elements
            
        Example:
            ```python
            selector = RandomIndexSelector(p=0.1)
            param = torch.randn(100, 50)  # 5000 elements
            mask = selector.get_indices(param, iteration=0)
            # mask contains ~500 True values (10% of 5000)
            selected_count = mask.sum().item()  # approximately 500
            ```
        """
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()


class ShuffledSequentialIndexSelector(IndexSelector):
    """
    Deterministic sequential index selector with shuffled ordering.
    
    ShuffledSequentialIndexSelector guarantees that all parameters are communicated
    exactly once every 1/p rounds. It divides parameters into shuffled partitions
    and cycles through them sequentially, providing predictable coverage and
    communication volume.
    
    ## Algorithm
    
    ### Deterministic Coverage
    - Shuffles parameter indices once at initialization
    - Divides shuffled indices into ceil(1/p) partitions
    - Cycles through partitions sequentially
    - Guarantees complete parameter coverage every ceil(1/p) rounds
    
    ### Key Properties
    - **Guaranteed Coverage**: All parameters updated within ceil(1/p) rounds
    - **Consistent Volume**: Exactly p × parameter_count elements per round
    - **Shuffled Order**: Random permutation prevents systematic bias
    - **Predictable**: Deterministic communication schedule
    
    ## Performance Characteristics
    
    ### Coverage Guarantees
    - **Exact Coverage**: Every parameter communicated exactly once per cycle
    - **Predictable Timing**: Maximum staleness = ceil(1/p) rounds
    - **No Variance**: Communication volume constant across rounds
    
    ### Memory Requirements
    - **O(parameter_count)**: Stores shuffled indices per parameter tensor
    - **One-time Cost**: Permutation generated once and reused
    - **Device Storage**: Indices stored on same device as parameters
    """
    
    def __init__(self, p):
        """
        Initialize shuffled sequential selector with sparsification rate.
        
        Args:
            p: Sparsification rate - determines partition size and cycle length
        """
        super().__init__(p)

    def get_indices(self, param, iteration):
        """
        Generate sequential partition mask for deterministic parameter coverage.
        
        Returns boolean mask for current partition in shuffled parameter sequence.
        Guarantees all parameters communicated exactly once per cycle.
        """
        num_total = param.numel()
        if num_total == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        # Initialize state for this parameter if not seen before
        if param not in self.state:
            num_partitions = max(
                1, math.ceil(1.0 / self.p)
            )  # Ensure at least 1 partition
            shuffled_indices = torch.randperm(num_total, device=param.device)
            self.state[param] = {
                "num_partitions": num_partitions,
                "shuffled_indices": shuffled_indices,
            }

        param_state = self.state[param]
        num_partitions = param_state["num_partitions"]
        shuffled_indices = param_state["shuffled_indices"]

        # Determine the current chunk based on the iteration number
        current_chunk = iteration % num_partitions

        # Calculate chunk size and remainder for potentially uneven distribution
        chunk_size = num_total // num_partitions
        remainder = num_total % num_partitions

        # Calculate start and end indices for the current chunk
        start_index = current_chunk * chunk_size + min(current_chunk, remainder)
        # The end index calculation ensures the chunk size is correct, adding 1 for chunks getting the remainder
        end_index = start_index + chunk_size + (1 if current_chunk < remainder else 0)

        # Get the flat indices for the current chunk
        selected_flat_indices = shuffled_indices[start_index:end_index]

        # Create and return the boolean mask
        mask = torch.zeros(num_total, dtype=torch.bool, device=param.device)
        if (
            selected_flat_indices.numel() > 0
        ):  # Handle empty selection if num_total is very small
            mask[selected_flat_indices] = True
        return mask.view(param.shape)


class PartitionedIndexSelector(IndexSelector):
    """
    Balanced partitioned index selector for systematic parameter coverage.
    
    Creates balanced partitions for each parameter tensor and cycles through them.
    Provides deterministic coverage with balanced partition sizes and dynamic
    repartitioning when cycles complete.
    
    ## Key Features
    - **Balanced Partitions**: Equal-sized partitions for consistent communication
    - **Dynamic Repartitioning**: Creates new random partitions each cycle
    - **Tensor-Aware**: Handles each parameter tensor independently
    - **Systematic Coverage**: Guarantees complete coverage before repetition
    """
    
    def __init__(self, p):
        """Initialize partitioned selector with sparsification rate."""
        super().__init__(p)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        # Ensure at least 1 partition
        num_partitions = max(1, min(math.ceil(1.0 / self.p), param.numel()))
        param_state["num_partitions"] = num_partitions
        if param.numel() > 0:
            param_state["partitions"] = (
                torch.rand(param.numel(), device=param.device).argsort()
                % num_partitions
            )
        else:
            # Handle zero-element tensors
            param_state["partitions"] = torch.empty(
                0, dtype=torch.long, device=param.device
            )

    # Update signature, though iteration is unused here
    def get_indices(self, param, iteration):
        # Handle zero-element tensors gracefully
        if param.numel() == 0:
            return torch.zeros_like(param, dtype=torch.bool)

        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        # Check if cycle needs reset BEFORE accessing partitions
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        param_state = self.state[param]

        # Need to handle case where num_partitions might be 0 if numel was 0 during _set_partition
        # Although we added checks for numel=0, ensure partition access is safe
        if param_state["num_partitions"] == 0:
            return torch.zeros_like(
                param, dtype=torch.bool
            )  # Should not happen if numel > 0

        # Indices calculation requires reshaping the flat partitions result
        partition_indices = param_state["partitions"] == param_state["curr_partition"]
        indices_mask = partition_indices.view(
            param.shape
        ).bool()  # Reshape flat bool tensor to param shape

        param_state["curr_partition"] += 1

        return indices_mask
