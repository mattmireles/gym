"""
ExoGym DeMo Strategy - Decoupled Momentum Optimization for Distributed Training

This module implements DeMo (Decoupled Momentum Optimization) strategy for distributed
training, providing advanced gradient compression and communication optimization through
DCT-based sparsification and momentum decoupling. DeMo represents cutting-edge research
in communication-efficient distributed training.

## DeMo Algorithm Overview

### Core Innovation: Decoupled Momentum
DeMo decouples momentum computation from gradient communication, enabling:
- **Local Momentum**: Momentum accumulated locally without communication overhead
- **Compressed Communication**: Only sparse, compressed gradient information transmitted
- **Maintained Convergence**: Theoretical guarantees for convergence preservation

### Key Technical Components
1. **DCT Compression**: Discrete Cosine Transform for gradient frequency analysis
2. **Top-K Sparsification**: Transmit only most significant frequency components
3. **Momentum Decoupling**: Separate local momentum from distributed updates
4. **Sign-SGD Integration**: Simplified gradient descent with sign quantization

## Research Foundation

### Academic Reference
- **Paper**: "DeMo: Decoupled Momentum Optimization" (Peng et al., 2024)
- **arXiv**: https://arxiv.org/abs/2411.19870
- **Implementation**: Based on official research code from https://github.com/bloc97/DeMo

### Theoretical Contributions
- **Communication Complexity**: Significant reduction in bandwidth requirements
- **Convergence Analysis**: Maintains convergence rates comparable to centralized training
- **Momentum Theory**: Novel approach to momentum in distributed settings

## Algorithm Components

### DCT-based Compression
- **Frequency Domain Transform**: Convert gradients to frequency representation
- **Sparse Encoding**: Retain only top-K frequency components
- **Adaptive Chunking**: Divide large tensors into manageable chunks for DCT
- **Lossless Reconstruction**: Exact reconstruction of transmitted components

### Momentum Decoupling
- **Local Delta**: Maintain momentum-like state locally
- **Compression Decay**: Exponential decay of uncompressed gradients
- **Transmission Estimation**: Predict what will be communicated for local correction
- **Residual Accumulation**: Build up residual for future transmission

## Performance Characteristics

### Communication Efficiency
- **Bandwidth Reduction**: Typically 10-100x reduction in communication volume
- **Adaptive Compression**: Compression rate adapts to gradient structure
- **Frequency Selectivity**: Focuses on most important gradient frequencies

### Computational Overhead
- **DCT Transform**: Additional computation for frequency domain conversion
- **Top-K Selection**: Efficient sparse selection algorithms
- **Momentum Management**: Local state tracking and updates

### Convergence Properties
- **Research Validated**: Empirical validation on large-scale models
- **Theoretical Foundation**: Convergence guarantees under specific conditions
- **Practical Performance**: Demonstrated effectiveness on real-world tasks

## Integration with ExoGym

### Strategy Pattern Compliance
- **Full Compatibility**: Implements Strategy interface for seamless integration
- **Device Support**: Works with CUDA, MPS, and CPU backends
- **Configuration System**: Full integration with logging and experiment tracking

### Hardware Considerations
- **Memory Requirements**: Additional memory for DCT transforms and compression state
- **Compute Requirements**: DCT transforms require additional GPU/CPU computation
- **Communication**: Custom all_gather integration for distributed coordination

## Current Status and Limitations

### Performance Notes
- **TODO Comment**: "This is really slow at the moment..." indicates optimization needed
- **Research Implementation**: Current version prioritizes correctness over performance
- **Production Readiness**: May require optimization for production deployment

### Known Issues
- **Performance Bottlenecks**: DCT transforms can be computationally expensive
- **Memory Usage**: Additional state tracking increases memory footprint
- **Numerical Stability**: Complex transform chains may introduce numerical issues

## Called by:
- Research environments requiring advanced compression techniques
- Bandwidth-constrained distributed training scenarios
- Experimental setups for communication optimization research

## Calls:
- DeMo optimizer implementation for core algorithm logic
- communicate.all_gather for distributed gradient synchronization
- Strategy base class for optimization and learning rate management

## Usage Patterns:

### Basic DeMo Training
```python
strategy = DeMoStrategy()  # Use default compression parameters
```

### Custom Compression Configuration
```python
strategy = DeMoStrategy(
    compression_decay=0.995,    # Slower decay for more accumulation
    compression_topk=64,        # More frequency components
    compression_chunk=128,      # Larger DCT chunks
    weight_decay=1e-4          # L2 regularization
)
```

### Research and Experimentation
```python
strategy = DeMoStrategy(
    compression_decay=0.999,
    compression_topk=16,        # Aggressive compression
    compression_chunk=32,       # Smaller chunks for fine-grained control
)
```

This implementation provides access to cutting-edge distributed training research
while maintaining compatibility with the ExoGym framework for experimentation
and validation.
"""

from .strategy import Strategy
from .communicate import all_gather

from .demo_impl.demo import DeMo


class DeMoStrategy(Strategy):
    """
    DeMo (Decoupled Momentum Optimization) distributed training strategy.
    
    DeMoStrategy implements the DeMo algorithm for communication-efficient distributed
    training using DCT-based gradient compression and momentum decoupling. This strategy
    represents cutting-edge research in bandwidth-constrained distributed training.
    
    ## Algorithm Innovation
    
    ### Decoupled Momentum Approach
    Traditional distributed training couples momentum computation with gradient communication,
    requiring frequent synchronization. DeMo decouples these processes:
    - **Local Momentum**: Maintained independently on each node
    - **Compressed Communication**: Only essential gradient information transmitted
    - **Residual Tracking**: Uncompressed gradients accumulated for future transmission
    
    ### DCT Compression Pipeline
    1. **Frequency Transform**: Convert gradients to DCT frequency domain
    2. **Sparsification**: Select top-K most significant frequency components
    3. **Compression**: Transmit only selected components across nodes
    4. **Reconstruction**: Reconstruct gradients from compressed representation
    
    ## Configuration Parameters
    
    ### Compression Settings
    - **compression_decay**: Controls residual accumulation rate (default: 0.999)
    - **compression_topk**: Number of frequency components to transmit (default: 32)
    - **compression_chunk**: DCT chunk size for transform efficiency (default: 64)
    
    ### Training Settings
    - **weight_decay**: L2 regularization strength (default: 0.0)
    
    ## Performance Trade-offs
    
    ### Communication Efficiency
    - **Bandwidth Reduction**: Significant reduction in communication volume
    - **Adaptive Compression**: Rate adapts to gradient frequency content
    - **Scalability**: Better scaling to large numbers of nodes
    
    ### Computational Overhead
    - **DCT Transforms**: Additional computation for frequency domain operations
    - **Compression Logic**: Top-K selection and reconstruction overhead
    - **State Management**: Additional memory for residual tracking
    
    Attributes:
        compression_decay: Exponential decay rate for uncompressed gradient residuals
        compression_topk: Number of top frequency components to communicate
        compression_chunk: Size of chunks for DCT transform processing
        weight_decay: L2 regularization coefficient
    """
    
    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_topk: int = 32,
        compression_chunk: int = 64,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Initialize DeMo strategy with compression and training configuration.
        
        Args:
            compression_decay: Exponential decay rate for gradient residuals (0 < decay < 1)
            compression_topk: Number of top frequency components to transmit per chunk
            compression_chunk: DCT chunk size for frequency domain transforms
            weight_decay: L2 regularization strength (default: 0.0)
            **kwargs: Additional strategy configuration (LR scheduling, etc.)
        """
        super().__init__(**kwargs)

        # Store DeMo algorithm configuration parameters
        self.compression_decay = compression_decay
        self.compression_topk = compression_topk
        self.compression_chunk = compression_chunk
        self.weight_decay = weight_decay

    def _init_node(self, model, rank, num_nodes):
        """
        Initialize DeMo strategy for distributed training node.
        
        This method sets up the DeMo optimizer with the configured compression
        parameters and integrates it with the ExoGym distributed training framework.
        The DeMo optimizer handles both local optimization and distributed communication
        within a single, integrated algorithm.
        
        ## Initialization Process
        
        ### DeMo Optimizer Configuration
        - **Compression Parameters**: DCT compression settings for bandwidth reduction
        - **Communication Integration**: Custom all_gather for ExoGym compatibility
        - **Device Setup**: Automatic handling of CUDA/MPS/CPU device placement
        
        ### ExoGym Integration
        - **Strategy Context**: Base strategy initialization for distributed training
        - **Scheduler Setup**: Learning rate scheduling configuration
        - **Logging Integration**: Communication volume tracking and experiment logging
        
        ## Communication Integration
        
        The DeMo optimizer requires a custom all_gather function for distributed
        communication. This integration ensures:
        - **Hardware Compatibility**: Works with ExoGym's MPS-compatible communication layer
        - **Error Handling**: Robust communication with automatic fallbacks
        - **Performance**: Optimized communication patterns for DeMo's specific needs
        
        Args:
            model: PyTorch model being trained (already on correct device)
            rank: Current node's rank in distributed training
            num_nodes: Total number of nodes participating in training
            
        Called by:
            Strategy initialization after model and distributed context are available
        """
        super()._init_node(model, rank, num_nodes)

        print("initialising DeMo engine")

        # Configure DeMo optimizer with compression and communication settings
        demo_kwargs = {
            "compression_decay": self.compression_decay,
            "compression_topk": self.compression_topk,
            "compression_chunk": self.compression_chunk,
            "weight_decay": self.weight_decay,
            "custom_all_gather": all_gather,  # ExoGym MPS-compatible communication
        }

        # Allow additional optimizer configuration via strategy config
        if hasattr(self, "strategy_config") and hasattr(
            self.strategy_config, "optimizer_kwargs"
        ):
            demo_kwargs.update(self.strategy_config.optimizer_kwargs)

        # Create DeMo optimizer instance with model parameters and configuration
        self.optim = DeMo(model.parameters(), **demo_kwargs)
        
        # Initialize learning rate scheduler for training
        self._setup_scheduler()

    def step(self):
        """
        Execute DeMo training step with integrated communication and optimization.
        
        DeMo integrates communication and optimization into a single step operation,
        unlike other strategies that separate gradient computation from communication.
        The DeMo optimizer handles the complete pipeline internally:
        
        ## DeMo Step Pipeline
        
        1. **Local Momentum Update**: Update local delta with current gradients
        2. **DCT Compression**: Transform and compress gradient residuals
        3. **Distributed Communication**: All-gather compressed gradient components
        4. **Gradient Reconstruction**: Reconstruct averaged gradients from compressed data
        5. **Sign-SGD Update**: Apply sign-based gradient descent to parameters
        
        ## Integration with ExoGym
        
        - **Communication Tracking**: DeMo optimizer tracks bytes communicated
        - **Step Management**: Base strategy handles LR scheduling and step counting
        - **Logging**: Communication metrics automatically logged via strategy framework
        
        Called by:
            TrainNode._train_step() after gradients are computed
            
        Calls:
            DeMo optimizer for integrated communication and optimization
            Strategy.step() for LR scheduling and communication logging
        """
        # DeMo optimizer handles complete communication and optimization pipeline
        self.optim.step()

        # Handle learning rate scheduling and communication volume logging
        super().step()
