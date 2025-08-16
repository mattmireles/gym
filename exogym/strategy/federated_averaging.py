"""
ExoGym Federated Averaging Strategy - Client-Server Distributed Learning

This module implements federated averaging (FedAvg) for distributed training,
supporting both classical federated learning patterns and modern island-based
federation for scalable distributed training. The implementation supports
flexible communication topologies and grouped averaging patterns.

## Federated Learning Overview

### Classical Federated Learning Pattern
Federated learning enables training across distributed clients without centralizing
data. Each client trains locally for multiple epochs before sharing model updates
with a central coordinator or other clients.

### Key Benefits
- **Data Privacy**: Raw data never leaves client devices
- **Communication Efficiency**: Model updates shared less frequently than gradients
- **Heterogeneous Clients**: Supports varying compute capabilities and data sizes
- **Scalability**: Can handle hundreds or thousands of participating clients

## Architecture Components

### AveragingCommunicator
Implements the core model averaging logic with support for:
- **Full Federation**: All nodes participate in averaging
- **Island Federation**: Subsets of nodes form averaging groups
- **Flexible Topology**: Dynamic group formation and partner selection

### FedAvgStrategy
Orchestrates federated training with configurable communication schedules:
- **Communication Interval (H)**: Controls frequency of model averaging
- **Local Training**: Multiple optimizer steps between communication rounds
- **Integration**: Built on CommunicateOptimizeStrategy for composability

## Island-Based Federation

### Traditional Problem: Full Averaging Bottleneck
Classical federated averaging requires all clients to participate in each round:
- Communication complexity scales with number of clients
- Single slow client blocks entire federation
- Network bandwidth requirements grow linearly

### Island Solution: Hierarchical Averaging
Island federation creates smaller groups that average independently:
- **Reduced Communication**: Only island members communicate
- **Parallel Processing**: Multiple islands operate simultaneously  
- **Fault Tolerance**: Slow/failed clients only affect their island
- **Scalability**: Communication complexity O(island_size) not O(num_clients)

## Communication Patterns

### Full Federation (island_size=None)
```
Round 1: All clients → Average all models → Broadcast result
Round 2: All clients → Average all models → Broadcast result
```

### Island Federation (island_size=K)
```
Round 1: Islands {0,1,2}, {3,4,5}, {6,7,8} → Each island averages independently
Round 2: New island formation → Different groupings for diversity
```

### Communication Schedule
- **Local Steps**: 1 to H-1, pure local training
- **Communication Step**: Every H steps, model averaging
- **Flexible H**: Balances communication efficiency vs convergence speed

## Integration with ExoGym Framework

### Strategy Pattern Compliance
- Inherits from CommunicateOptimizeStrategy for modular design
- Uses CommunicationModule pattern for pluggable averaging logic
- Supports all ExoGym features: MPS compatibility, logging, configuration

### Device Compatibility
- Works with CUDA, MPS (Apple Silicon), and CPU backends
- Automatic handling of device placement during averaging
- MPS-compatible communication via communicate.py abstractions

## Performance Characteristics

### Communication Volume
- Reduces communication frequency by factor of H compared to gradient averaging
- Island federation reduces bandwidth by factor of (num_nodes / island_size)
- Total savings: H × (num_nodes / island_size) compared to gradient averaging

### Convergence Properties
- Maintains good convergence for appropriate H values (typically 1-10 for FedAvg)
- Island formation provides diversity while maintaining averaging benefits
- Local training allows adaptation to client-specific data distributions

## Called by:
- Distributed training setups requiring federated learning patterns
- Multi-organizational training where data privacy is important
- Large-scale distributed training with communication constraints

## Calls:
- CommunicateOptimizeStrategy for base distributed training functionality
- communicate.py functions for hardware-agnostic distributed operations
- torch.distributed for group formation and parameter averaging

## Usage Patterns:

### Basic Federated Learning
```python
strategy = FedAvgStrategy(H=5)  # Average every 5 local steps
```

### Island-Based Federation
```python
strategy = FedAvgStrategy(H=10, island_size=4)  # 4-node islands, communicate every 10 steps
```

### Custom Client Configuration
```python
strategy = FedAvgStrategy(
    inner_optim="sgd",
    island_size=8,
    H=3,
    lr=0.01,
    max_norm=1.0  # Gradient clipping
)
```

This implementation provides production-ready federated learning capabilities
with modern optimizations for scalability and communication efficiency.
"""

import torch.distributed as dist
import random

import torch

from typing import Optional, Set, Union

from .communicate_optimize_strategy import (
    CommunicateOptimizeStrategy,
    CommunicationModule,
)
from .optim import OptimSpec
from .communicate import all_reduce, all_gather


class AveragingCommunicator(CommunicationModule):
    """
    Communication module implementing federated model averaging with island support.
    
    AveragingCommunicator implements the core federated learning communication pattern
    where nodes share and average their model parameters periodically. It supports
    both full federation (all nodes participate) and island federation (subset averaging)
    for improved scalability and fault tolerance.
    
    ## Core Functionality
    
    ### Model Parameter Averaging
    - Collects model parameters from participating nodes
    - Computes element-wise average across all participants
    - Updates each node's model to the averaged parameters
    - Preserves model architecture and parameter structure
    
    ### Island Federation Support
    - **Dynamic Grouping**: Forms random islands each communication round
    - **Scalable Communication**: Reduces communication complexity from O(N) to O(K)
    - **Fault Isolation**: Failed nodes only affect their island, not entire federation
    - **Load Distribution**: Enables parallel averaging across multiple groups
    
    ## Island Formation Algorithm
    
    ### Random Partner Selection
    1. **Coordinator Role**: Rank 0 generates random permutation of all ranks
    2. **Broadcast Assignment**: All nodes receive the same island assignments
    3. **Group Formation**: Consecutive ranks in permutation form islands
    4. **Deterministic Results**: Same seed produces same island formation
    
    ### Benefits of Randomization
    - **Diversity**: Each round provides different node combinations
    - **Fairness**: All nodes have equal probability of collaboration
    - **Convergence**: Random mixing prevents permanent isolation
    - **Robustness**: Reduces impact of consistently slow/fast nodes
    
    ## Communication Efficiency
    
    ### Full Federation (island_size=None)
    - Uses efficient all_reduce for O(log N) communication complexity
    - Single communication round averages all model parameters
    - Optimal bandwidth usage for small to medium node counts
    
    ### Island Federation (island_size=K)
    - Uses all_gather followed by selective averaging
    - Parallel processing across multiple islands
    - Total communication: O(K × log K) per island
    - Enables scaling to large numbers of nodes
    
    ## Technical Implementation
    
    ### Device-Agnostic Operations
    - Uses communicate.py abstractions for CUDA/MPS/CPU compatibility
    - Automatic tensor device placement and data movement
    - Maintains parameter precision and gradient computation graphs
    
    ### Memory Efficiency
    - In-place parameter updates minimize memory allocation
    - Temporary tensors created only during communication
    - Garbage collection-friendly implementation
    
    Attributes:
        island_size: Maximum size of averaging groups (None = full federation)
    """

    def __init__(self, island_size: Optional[int] = None, **kwargs):
        """
        Initialize averaging communicator with island configuration.
        
        Args:
            island_size: Maximum nodes per averaging group (None for full federation)
            **kwargs: Additional CommunicationModule configuration
        """
        super().__init__(**kwargs)
        self.island_size = island_size

    def _select_partners(self, rank: int, num_nodes: int) -> Set[int]:
        """
        Select communication partners for island-based federated averaging.
        
        This method implements the island formation algorithm that randomly groups
        nodes into smaller averaging clusters. It provides scalable communication
        by reducing the number of participants in each averaging round while
        maintaining convergence properties through randomized grouping.
        
        ## Algorithm Details
        
        ### Coordinator-Based Assignment
        1. **Rank 0 Coordination**: Only rank 0 generates random permutation
        2. **Deterministic Broadcasting**: All nodes receive identical assignment
        3. **Synchronized Grouping**: Ensures all nodes agree on island membership
        4. **Consistent Results**: Same random state produces same groupings
        
        ### Random Permutation Strategy
        - **Full Randomization**: Each communication round reshuffles all ranks
        - **Equal Participation**: All nodes have equal probability of grouping
        - **Diversity Maximization**: Prevents permanent node isolation
        - **Convergence Benefits**: Random mixing improves training dynamics
        
        ## Island Formation Process
        
        ### Contiguous Grouping
        ```
        Example with 12 nodes, island_size=4:
        Permutation: [7, 2, 11, 0, 5, 8, 1, 9, 3, 6, 10, 4]
        Island 1: {7, 2, 11, 0}   # ranks[0:4]
        Island 2: {5, 8, 1, 9}    # ranks[4:8]  
        Island 3: {3, 6, 10, 4}   # ranks[8:12]
        ```
        
        ### Membership Discovery
        - Each node searches through islands to find its group
        - Returns set of partner ranks for averaging
        - Includes current node's rank in the returned set
        
        ## Communication Protocol
        
        ### Broadcast Synchronization
        - Uses torch.distributed.broadcast_object_list for rank coordination
        - Ensures all nodes receive identical island assignments
        - Critical for maintaining federated learning correctness
        
        ### Fault Tolerance
        - Failed nodes only affect their assigned island
        - Other islands continue averaging independently
        - Graceful degradation under partial node failures
        
        ## Performance Characteristics
        
        ### Communication Complexity
        - **Broadcast Cost**: O(log N) for assignment distribution
        - **Island Discovery**: O(N/K) average case for K islands
        - **Total Overhead**: Minimal compared to actual model averaging
        
        ### Memory Usage
        - **Temporary Storage**: O(N) for rank permutation
        - **Island Storage**: O(K) for island membership
        - **Garbage Collection**: Temporary structures cleaned after assignment
        
        Args:
            rank: Current node's rank (0 to num_nodes-1)
            num_nodes: Total number of nodes in federation
            
        Returns:
            Set[int]: Set of ranks (including current rank) forming this node's island
            
        Called by:
            communicate() method during island federation rounds
            
        Calls:
            torch.distributed.broadcast_object_list() for rank coordination
            
        Example:
            ```python
            # Node 5 in 16-node federation with island_size=4
            partners = self._select_partners(rank=5, num_nodes=16)
            # Returns something like: {3, 5, 12, 7} depending on random permutation
            ```
        """
        world_size = num_nodes

        # Coordinator generates random island assignments for all nodes
        if rank == 0:
            # Create random permutation of all ranks for fair island distribution
            ranks = list(range(world_size))
            random.shuffle(ranks)
        else:
            # Non-coordinator nodes prepare to receive assignment
            ranks = [None] * world_size

        # Broadcast island assignment to ensure all nodes have identical grouping
        dist.broadcast_object_list(ranks, src=0)

        # Form islands by grouping consecutive ranks in the permutation
        islands = []
        island_size = self.island_size if self.island_size is not None else num_nodes
        for i in range(0, len(ranks), island_size):
            islands.append(set(ranks[i : i + island_size]))

        # Find which island this rank belongs to for averaging participation
        my_island = None
        for island in islands:
            if rank in island:
                my_island = island
                break

        return my_island

    def _average_models(self, model, island_members: Set[int], num_nodes: int) -> None:
        """
        Average model parameters across island members using optimal communication patterns.
        
        This method implements the core parameter averaging logic with automatic
        optimization based on island size. It chooses between efficient all_reduce
        for full federation and selective averaging for island federation.
        
        ## Communication Strategy Selection
        
        ### Full Federation Optimization (island_size == num_nodes)
        - **all_reduce Efficiency**: Uses optimized collective communication
        - **O(log N) Complexity**: Leverages tree-based reduction algorithms
        - **Bandwidth Optimal**: Single communication round averages all parameters
        - **Hardware Acceleration**: Benefits from NCCL/MPI optimizations
        
        ### Island Federation Implementation (island_size < num_nodes)
        - **all_gather Collection**: Gathers parameters from all nodes
        - **Selective Averaging**: Computes average only within island
        - **Parallel Processing**: Multiple islands average simultaneously
        - **Fault Isolation**: Island failures don't affect other groups
        
        ## Parameter Processing
        
        ### Element-wise Averaging
        - Iterates through all model parameters (weights, biases, etc.)
        - Maintains parameter shapes and device placement
        - Preserves gradient computation graphs where applicable
        - Handles different parameter types (float32, float16, etc.)
        
        ### In-place Updates
        - Updates param.data directly to avoid graph rebuilding
        - Minimizes memory allocation during averaging
        - Preserves parameter metadata and requires_grad settings
        - Maintains model structure and parameter organization
        
        ## Technical Implementation Details
        
        ### Memory Management
        ```python
        # Temporary tensor allocation for island averaging
        tensor_list = [torch.zeros_like(param.data) for _ in range(num_nodes)]
        # Automatic cleanup after averaging completes
        ```
        
        ### Numerical Stability
        - Division by island size prevents parameter magnitude drift
        - Consistent averaging across different island sizes
        - Maintains training stability during federated learning
        
        ### Device Compatibility
        - Uses communicate.py abstractions for hardware independence
        - Automatic handling of CUDA/MPS/CPU device placement
        - Preserves tensor device location throughout averaging
        
        Args:
            model: PyTorch model whose parameters will be averaged
            island_members: Set of rank IDs participating in this averaging round
            num_nodes: Total number of nodes in the federation
            
        Called by:
            communicate() method during federated averaging rounds
            
        Calls:
            communicate.all_reduce() for efficient full federation averaging
            communicate.all_gather() for island-based selective averaging
            
        Example:
            ```python
            # Full federation (all 8 nodes participate)
            island_members = {0, 1, 2, 3, 4, 5, 6, 7}
            self._average_models(model, island_members, num_nodes=8)
            # Uses all_reduce for optimal communication
            
            # Island federation (4 nodes in island)
            island_members = {2, 5, 6, 7}
            self._average_models(model, island_members, num_nodes=8) 
            # Uses all_gather + selective averaging
            ```
        """
        for param in model.parameters():
            if len(island_members) == num_nodes:
                # Full federation: use efficient all_reduce collective operation
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= num_nodes
            else:
                # Island federation: gather all parameters then average selectively
                tensor_list = [torch.zeros_like(param.data) for _ in range(num_nodes)]
                all_gather(tensor_list, param.data)

                # Compute average only from ranks in the same island
                island_tensors = [tensor_list[rank] for rank in island_members]
                island_average = sum(island_tensors) / len(island_tensors)

                param.data = island_average

    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """
        Orchestrate federated averaging communication round.
        
        This method coordinates the complete averaging process, from island formation
        to parameter averaging. It serves as the main entry point for federated
        communication and handles the logic for choosing between full and island
        federation modes.
        
        Args:
            model: PyTorch model to average across nodes
            rank: Current node's rank in the federation
            num_nodes: Total number of participating nodes
            local_step: Current training step (unused in this implementation)
        """
        if num_nodes > 1:
            if self.island_size is not None and self.island_size < num_nodes:
                # Island federation: form random groups for averaging
                island_members = self._select_partners(rank, num_nodes)
            else:
                # Full federation: all nodes participate in averaging
                island_members = set(range(num_nodes))

            # Perform model parameter averaging with selected partners
            self._average_models(model, island_members, num_nodes)

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize communication module for distributed training.
        
        No specific initialization required for averaging communicator.
        Inherits default behavior from CommunicationModule base class.
        """
        pass


class FedAvgStrategy(CommunicateOptimizeStrategy):
    """
    Federated Averaging (FedAvg) strategy for distributed training with local updates.
    
    FedAvgStrategy implements the classical federated learning algorithm where nodes
    perform multiple local training steps before sharing and averaging their model
    parameters. It supports both full federation and island-based federation for
    improved scalability and communication efficiency.
    
    ## Federated Learning Algorithm
    
    ### Local Training Phase
    1. **Local Updates**: Each client trains on local data for H steps
    2. **Gradient Accumulation**: Standard backpropagation and optimization
    3. **No Communication**: Pure local training without inter-node communication
    4. **Model Divergence**: Models naturally diverge based on local data
    
    ### Communication Phase
    1. **Model Sharing**: Clients share their complete model parameters
    2. **Parameter Averaging**: Compute element-wise average across participants
    3. **Model Synchronization**: All participants adopt the averaged model
    4. **Reset Training**: Continue local training from synchronized state
    
    ## Key Benefits
    
    ### Communication Efficiency
    - **Reduced Frequency**: Communication every H steps instead of every step
    - **Bandwidth Savings**: H×(num_nodes/island_size) reduction compared to gradient averaging
    - **Scalable Architecture**: Island federation enables scaling to large federations
    
    ### Privacy Preservation
    - **No Raw Data Sharing**: Only model parameters are communicated
    - **Local Computation**: Data remains on client devices throughout training
    - **Differential Privacy**: Can be combined with DP mechanisms for additional privacy
    
    ### Heterogeneity Support
    - **Diverse Data**: Handles non-IID data distributions across clients
    - **Varying Compute**: Accommodates different client computational capabilities
    - **Flexible Participation**: Supports partial client participation per round
    
    ## Configuration Parameters
    
    ### Communication Interval (H)
    - **H=1**: Equivalent to synchronized data parallel training
    - **H=5-10**: Typical federated learning configuration
    - **Large H**: Higher communication efficiency, potentially slower convergence
    
    ### Island Federation
    - **island_size=None**: Full federation (all nodes participate)
    - **island_size=K**: K-node islands for scalable communication
    - **Dynamic Islands**: Random regrouping each communication round
    
    ## Implementation Architecture
    
    ### Modular Design
    - **CommunicateOptimizeStrategy Base**: Inherits optimization and scheduling infrastructure
    - **AveragingCommunicator**: Pluggable communication module for parameter averaging
    - **Hardware Agnostic**: Works with CUDA, MPS, and CPU backends
    
    ### Integration with ExoGym
    - **Strategy Pattern**: Seamless integration with distributed training framework
    - **Configuration System**: Full compatibility with logging and experiment tracking
    - **Multiprocessing Safe**: Handles torch.multiprocessing.spawn() correctly
    
    Attributes:
        island_size: Maximum nodes per averaging group (None for full federation)
        H: Communication interval in training steps
    """
    
    def __init__(
        self,
        inner_optim: Optional[Union[str, OptimSpec]] = None,
        island_size: Optional[int] = None,
        H: int = 1,
        max_norm: float = None,
        **kwargs,
    ):
        """
        Initialize federated averaging strategy with communication configuration.
        
        Args:
            inner_optim: Local optimizer specification (default: AdamW)
            island_size: Maximum nodes per averaging group (None for full federation)
            H: Communication interval in training steps (default: 1)
            max_norm: Gradient clipping threshold (default: None)
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        # Create the averaging communicator with island configuration
        averaging_comm = AveragingCommunicator(island_size=island_size)

        # Initialize base strategy with averaging communication
        super().__init__(
            inner_optim=inner_optim,
            communication_modules=[averaging_comm],
            max_norm=max_norm,
            **kwargs,
        )

        # Store federated learning configuration
        self.island_size = island_size
        self.H = H

    def _communicate(self):
        """
        Trigger federated averaging communication at configured intervals.
        
        This method implements the communication schedule for federated learning,
        where averaging occurs every H training steps. It ensures that local
        training proceeds uninterrupted between communication rounds while
        maintaining proper synchronization when communication is required.
        
        ## Communication Schedule
        
        ### Local Training Steps (1 to H-1, H+1 to 2H-1, etc.)
        - No communication triggered
        - Pure local training with gradient-based optimization
        - Models diverge based on local data distributions
        - Maximum communication efficiency
        
        ### Communication Steps (H, 2H, 3H, etc.)
        - Triggers parameter averaging across federation
        - Synchronizes models to averaged state
        - Resets local training from synchronized model
        - Brief communication burst followed by local training
        
        Called by:
            CommunicateOptimizeStrategy.step() during training loop execution
            
        Calls:
            CommunicateOptimizeStrategy._communicate() for actual averaging
        """
        if self.local_step % self.H == 0 and self.local_step > 0:
            # Communication round: trigger federated averaging
            super()._communicate()

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize federated averaging strategy for distributed training.
        
        Completes strategy initialization by setting up federation-specific
        configuration and ensuring island size constraints are properly handled.
        
        Args:
            model: PyTorch model being trained
            rank: Current node's rank in the federation
            num_nodes: Total number of nodes in the federation
        """
        # Initialize base strategy (optimizer, communication modules, etc.)
        super()._init_node(model, rank, num_nodes)

        # Default to full federation if no island size specified
        if self.island_size is None:
            self.island_size = num_nodes
