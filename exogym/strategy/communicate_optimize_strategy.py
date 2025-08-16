"""
ExoGym CommunicateOptimizeStrategy - Modular Communication and Optimization Framework

This module provides a flexible framework for building distributed training strategies
that combine local optimization with pluggable communication patterns. It enables
composition of complex distributed algorithms through modular communication components
and standardized optimization interfaces.

## Architecture Overview

### Modular Design Philosophy
The CommunicateOptimizeStrategy framework separates distributed training into two
orthogonal concerns:
- **Local Optimization**: Standard gradient-based parameter updates
- **Communication**: Distributed coordination and parameter synchronization

This separation enables:
- **Composability**: Mix and match different communication strategies
- **Reusability**: Share communication modules across different algorithms
- **Testability**: Unit test communication and optimization logic independently
- **Extensibility**: Add new communication patterns without modifying optimization

### Strategy Pattern Implementation
The framework implements the strategy pattern at two levels:
1. **CommunicationModule Strategy**: Pluggable communication algorithms
2. **CommunicateOptimizeStrategy**: Composition of communication modules with optimization

## Communication Module Framework

### Abstract Interface
CommunicationModule defines a standard interface for distributed communication:
- **communicate()**: Core communication logic called during training
- **_init_node()**: Node-specific initialization for distributed context
- **Stateful Design**: Modules can maintain internal state across communication rounds

### Pluggable Architecture
Multiple communication modules can be composed within a single strategy:
- **Sequential Execution**: Modules execute in specified order
- **Shared Context**: All modules operate on the same model and training state
- **Independent Logic**: Each module implements its own communication pattern

## Integration with ExoGym Framework

### Strategy Base Class Compliance
- **Full Compatibility**: Inherits all Strategy functionality (LR scheduling, logging, etc.)
- **Optimizer Integration**: Seamless integration with OptimSpec factory pattern
- **Device Support**: Works with CUDA, MPS, and CPU backends via communicate.py

### Used by Advanced Strategies
This framework serves as the foundation for sophisticated distributed algorithms:
- **FedAvgStrategy**: Uses AveragingCommunicator for federated parameter averaging
- **SPARTAStrategy**: Uses SparseCommunicator for bandwidth-efficient communication
- **Custom Strategies**: Enables rapid development of novel distributed algorithms

## Performance and Scalability

### Minimal Overhead
- **Direct Module Calls**: No reflection or dynamic dispatch overhead
- **Efficient Composition**: Linear execution of communication modules
- **Memory Efficient**: Shared model state across all communication modules

### Scalable Design
- **Stateless Modules**: Enable horizontal scaling of communication patterns
- **Asynchronous Compatible**: Framework supports async communication module development
- **Hardware Agnostic**: Abstract interface enables hardware-specific optimizations

## Called by:
- Advanced distributed training strategies requiring modular communication
- Researchers implementing novel distributed algorithms
- Production systems needing flexible communication patterns

## Calls:
- Base Strategy class for optimization and learning rate management
- CommunicationModule implementations for distributed coordination
- OptimSpec factory for flexible optimizer construction

## Usage Patterns:

### Single Communication Module
```python
averaging_comm = AveragingCommunicator(island_size=4)
strategy = CommunicateOptimizeStrategy(
    communication_modules=[averaging_comm],
    inner_optim="adamw"
)
```

### Multiple Communication Modules
```python
sparse_comm = SparseCommunicator(RandomIndexSelector(0.01))
compression_comm = CompressionCommunicator(compression_ratio=0.1)
strategy = CommunicateOptimizeStrategy(
    communication_modules=[sparse_comm, compression_comm],
    inner_optim="sgd"
)
```

### Custom Strategy Development
```python
class NovelStrategy(CommunicateOptimizeStrategy):
    def _communicate(self):
        if self.local_step % self.H == 0:
            super()._communicate()  # Apply all modules conditionally
```

This framework enables rapid development and testing of distributed training algorithms
while maintaining compatibility with the broader ExoGym ecosystem.
"""

import torch
from torch.nn import utils as nn_utils
from typing import List, Optional, Union
from abc import ABC, abstractmethod

from .strategy import Strategy
from .optim import OptimSpec, ensure_optim_spec


class CommunicationModule(ABC):
    """
    Abstract base class defining the interface for pluggable communication modules.
    
    CommunicationModule enables the strategy pattern for distributed communication,
    allowing different algorithms to be composed and reused across training strategies.
    Each module encapsulates a specific communication pattern (averaging, sparsification,
    compression, etc.) and can be combined with others for complex distributed algorithms.
    
    ## Design Principles
    
    ### Single Responsibility
    Each communication module implements one specific communication pattern:
    - **AveragingCommunicator**: Parameter averaging for federated learning
    - **SparseCommunicator**: Sparse parameter communication for bandwidth reduction
    - **CompressionCommunicator**: Parameter compression for network efficiency
    
    ### Stateful Operation
    Modules can maintain state across communication rounds:
    - **Iteration Counters**: For temporal communication patterns
    - **Partner Selection**: For dynamic topology algorithms
    - **Compression State**: For adaptive compression schemes
    
    ### Hardware Abstraction
    Modules use communicate.py abstractions for device independence:
    - **CUDA/MPS/CPU**: Transparent operation across hardware backends
    - **Automatic Device Handling**: Parameters remain on correct devices
    - **Memory Efficiency**: Minimal overhead for communication operations
    
    ## Interface Requirements
    
    Subclasses must implement three methods for complete communication functionality:
    - **__init__()**: Module-specific configuration and state initialization
    - **communicate()**: Core communication logic executed during training
    - **_init_node()**: Node-specific setup after distributed context is available
    
    ## Integration Patterns
    
    ### Strategy Composition
    Multiple communication modules can be composed within a single strategy:
    ```python
    strategy = CommunicateOptimizeStrategy(
        communication_modules=[sparse_comm, compression_comm, averaging_comm]
    )
    ```
    
    ### Execution Order
    Modules execute sequentially in the order specified, enabling:
    - **Pipeline Patterns**: Sparsification → Compression → Averaging
    - **Layered Communication**: Multiple algorithms operating on same data
    - **Conditional Execution**: Different modules for different phases
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize communication module with algorithm-specific configuration.
        
        Subclasses should configure their communication parameters and initialize
        any state required for their specific communication pattern.
        """
        pass

    @abstractmethod
    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """
        Execute communication algorithm for the given model and distributed context.
        
        This method contains the core communication logic and is called during
        training when the strategy determines communication should occur.
        
        ## Implementation Guidelines
        
        ### Model Parameter Handling
        - Operate directly on model.parameters() or model.named_parameters()
        - Preserve parameter shapes, devices, and gradient computation graphs
        - Use torch.no_grad() context for parameter modifications
        
        ### Distributed Coordination
        - Use communicate.py functions for hardware-agnostic operations
        - Ensure all nodes execute identical communication patterns
        - Handle synchronization and collective operations properly
        
        ### State Management
        - Update internal state (iteration counters, partner selection, etc.)
        - Maintain consistency across communication rounds
        - Handle initialization of per-parameter state as needed
        
        Args:
            model: PyTorch model whose parameters will be communicated
            rank: Current node's rank in distributed training (0 to num_nodes-1)
            num_nodes: Total number of nodes participating in training
            local_step: Current training step count for temporal patterns
            
        Called by:
            CommunicateOptimizeStrategy._communicate() during training execution
        """
        pass

    @abstractmethod
    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """
        Initialize communication module for specific distributed training node.
        
        This method is called after the model and distributed context are available,
        enabling node-specific setup that requires knowledge of the model structure,
        device placement, and distributed configuration.
        
        ## Initialization Tasks
        
        ### Model-Dependent Setup
        - Analyze model structure for communication optimization
        - Initialize per-parameter state dictionaries
        - Configure communication buffers based on model size
        
        ### Distributed Context Setup
        - Configure rank-specific behavior (coordinator vs. participant roles)
        - Initialize communication groups or topologies
        - Set up device-specific optimizations
        
        Args:
            model: PyTorch model being trained (already on correct device)
            rank: Current node's rank in distributed training
            num_nodes: Total number of nodes in the training federation
            
        Called by:
            CommunicateOptimizeStrategy._init_node() during strategy initialization
        """
        pass


class CommunicateOptimizeStrategy(Strategy):
    """
    Flexible framework for distributed strategies combining local optimization with modular communication.
    
    CommunicateOptimizeStrategy provides a foundation for building sophisticated distributed
    training algorithms by composing local optimization with pluggable communication modules.
    This design enables rapid development of novel distributed algorithms while maintaining
    compatibility with the broader ExoGym framework.
    
    ## Architecture Design
    
    ### Separation of Concerns
    The strategy cleanly separates two fundamental aspects of distributed training:
    1. **Local Optimization**: Standard gradient-based parameter updates using PyTorch optimizers
    2. **Distributed Communication**: Coordination between nodes via pluggable communication modules
    
    ### Modular Communication
    Communication is implemented through composable modules that can be mixed and matched:
    - **Single Module**: Simple algorithms like federated averaging
    - **Multiple Modules**: Complex pipelines like sparsification + compression + averaging
    - **Custom Timing**: Subclasses control when communication occurs
    
    ## Key Features
    
    ### Framework Benefits
    - **Rapid Prototyping**: New distributed algorithms can be built by composing existing modules
    - **Code Reuse**: Communication modules shared across different strategies
    - **Testing**: Optimization and communication logic can be unit tested independently
    - **Maintenance**: Changes to communication patterns don't affect optimization logic
    
    ### Production Ready
    - **Performance**: Minimal overhead from modular design
    - **Scalability**: Efficient execution of multiple communication modules
    - **Reliability**: Battle-tested optimization framework with pluggable communication
    - **Monitoring**: Full integration with ExoGym logging and configuration systems
    
    ## Communication Module Composition
    
    ### Sequential Execution
    Communication modules execute in the order specified, enabling sophisticated patterns:
    ```python
    # Pipeline: Sparsify → Compress → Average
    modules = [
        SparseCommunicator(RandomIndexSelector(0.01)),
        CompressionCommunicator(compression_ratio=0.1),
        AveragingCommunicator(island_size=4)
    ]
    ```
    
    ### Shared State
    All modules operate on the same model and have access to:
    - **Model Parameters**: Current parameter values and gradients
    - **Training Context**: Rank, number of nodes, current step
    - **Strategy State**: Access to parent strategy for coordination
    
    ## Integration with ExoGym
    
    ### Strategy Compliance
    - **Full Inheritance**: All Strategy features (LR scheduling, logging, configuration)
    - **OptimSpec Integration**: Flexible optimizer specification and construction
    - **Device Support**: Works with CUDA, MPS, and CPU backends
    - **Multiprocessing Safe**: Proper serialization and initialization patterns
    
    ### Used by Advanced Algorithms
    This framework serves as the foundation for sophisticated distributed strategies:
    - **FedAvgStrategy**: Federated learning with parameter averaging
    - **SPARTAStrategy**: Sparse communication for bandwidth efficiency
    - **Custom Research**: Novel algorithms built by composing communication patterns
    
    Attributes:
        communication_modules: List of communication modules executed sequentially
        inner_optim_spec: Optimizer specification for local parameter updates
        max_norm: Optional gradient clipping threshold for training stability
    """

    def __init__(
        self,
        communication_modules: List[CommunicationModule],
        inner_optim: Optional[Union[str, OptimSpec]] = None,
        max_norm: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize communication-optimization strategy with modular communication.
        
        Args:
            communication_modules: List of communication modules to execute sequentially
            inner_optim: Local optimizer specification (default: AdamW)
            max_norm: Gradient clipping threshold (default: None, no clipping)
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        super().__init__(**kwargs)

        # Configure local optimizer with flexible specification
        self.inner_optim_spec = ensure_optim_spec(inner_optim) or OptimSpec(
            torch.optim.AdamW
        )

        # Store communication modules for sequential execution
        self.communication_modules = communication_modules
        self.max_norm = max_norm

        # Provide strategy reference to communication modules for coordination
        for comm_module in self.communication_modules:
            comm_module.strategy = self

    def step(self):
        """
        Execute one training step with local optimization and communication.
        
        This method implements the core training loop pattern that combines local
        gradient-based optimization with distributed communication. The execution
        order ensures gradients are properly clipped, parameters are updated locally,
        and then communication occurs according to the strategy's schedule.
        
        ## Step Execution Flow
        
        1. **Gradient Clipping**: Apply gradient norm clipping if configured
        2. **Local Optimization**: Update parameters using local gradients
        3. **Communication**: Execute communication modules (strategy-dependent timing)
        4. **Base Step Processing**: Handle LR scheduling and step counting
        
        ## Gradient Clipping Integration
        Gradient clipping (if enabled) occurs before parameter updates to:
        - **Prevent Explosion**: Limit gradient norms for training stability
        - **Consistent Scaling**: Apply same clipping across all nodes
        - **Pre-Communication**: Clip before any communication occurs
        
        Called by:
            TrainNode._train_step() after gradients are computed
            
        Calls:
            Local optimizer for parameter updates
            _communicate() for distributed coordination
            Strategy.step() for LR scheduling and step tracking
        """
        # Apply gradient clipping for training stability if configured
        if self.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)

        # Execute local optimization step with computed gradients
        self.optim.step()

        # Execute communication modules according to strategy-specific timing
        self._communicate()

        # Handle learning rate scheduling and step counter management
        super().step()

    def _communicate(self):
        """
        Execute all communication modules sequentially.
        
        This method applies all configured communication modules in the order
        specified during initialization. Subclasses can override this method
        to implement custom communication timing and conditional logic.
        
        ## Sequential Execution Pattern
        Communication modules execute in order, enabling sophisticated patterns:
        - **Pipeline Processing**: Each module can modify the model state
        - **Layered Communication**: Multiple algorithms can operate on same data
        - **Dependency Chains**: Later modules can depend on earlier modifications
        
        ## Override Patterns for Custom Timing
        ```python
        class CustomStrategy(CommunicateOptimizeStrategy):
            def _communicate(self):
                if self.local_step % self.communication_interval == 0:
                    super()._communicate()  # Apply all modules conditionally
        ```
        
        Called by:
            step() method during each training iteration
            
        Calls:
            CommunicationModule.communicate() for each configured module
        """
        for comm_module in self.communication_modules:
            comm_module.communicate(
                self.model, self.rank, self.num_nodes, self.local_step
            )

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize strategy and all communication modules for distributed training.
        
        This method completes the strategy setup by initializing the distributed
        context, communication modules, optimizer, and learning rate scheduler.
        It ensures all components are properly configured for the specific node
        and distributed training environment.
        
        ## Initialization Sequence
        
        1. **Base Strategy Setup**: Initialize distributed context and step tracking
        2. **Communication Module Setup**: Initialize each module with distributed context
        3. **Optimizer Creation**: Build optimizer with model parameters on correct device
        4. **Scheduler Setup**: Configure learning rate scheduling
        
        Args:
            model: PyTorch model being trained (already on correct device)
            rank: Current node's rank in distributed training
            num_nodes: Total number of nodes participating in training
            
        Called by:
            Strategy initialization after model and distributed context are available
        """
        # Initialize base strategy with distributed training context
        super()._init_node(model, rank, num_nodes)

        # Initialize all communication modules with distributed context
        for comm_module in self.communication_modules:
            comm_module._init_node(model, rank, num_nodes)

        # Create optimizer instance bound to model parameters on correct device
        self.optim = self.inner_optim_spec.build(model)
        
        # Configure learning rate scheduler for training
        self._setup_scheduler()
