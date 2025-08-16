"""
ExoGym Strategy Module - Distributed Training Communication Strategies

This module provides a comprehensive collection of distributed training strategies
that define how nodes communicate and coordinate during training. Each strategy
implements different algorithms and communication patterns optimized for various
scenarios and constraints.

## Strategy Categories

### Core Strategies
- **Strategy**: Abstract base class defining the strategy interface
- **SimpleReduceStrategy**: Traditional gradient averaging (data parallel)
- **DiLoCoStrategy**: Distributed Low-Communication training with periodic model averaging

### Advanced Communication Strategies
- **SPARTAStrategy**: Sparsification-based communication reduction
- **CommunicateOptimizeStrategy**: Communication-optimized distributed training
- **DeMoStrategy**: Demonstration strategy for algorithm research

### Federated Learning
- **FedAvgStrategy**: Federated averaging for client-server training patterns

### Hybrid Strategies
- **SPARTADiLoCoStrategy**: Combines SPARTA sparsification with DiLoCo communication

## Strategy Selection Guide

### Communication Frequency vs. Convergence
- **High Communication**: SimpleReduceStrategy (every step)
- **Medium Communication**: DiLoCoStrategy (every H steps)
- **Low Communication**: SPARTA variants (sparse communication)

### Use Case Optimization
- **Fast Networks**: SimpleReduceStrategy for maximum convergence speed
- **Slow Networks**: DiLoCoStrategy or SPARTA for bandwidth efficiency
- **Federated Learning**: FedAvgStrategy for client-server scenarios
- **Research**: DeMoStrategy for algorithm experimentation

### Hardware Compatibility
All strategies support:
- **CUDA**: Full GPU acceleration with NCCL backend
- **MPS**: Apple Silicon with CPU fallback via communicate.py
- **CPU**: Gloo backend for CPU-only distributed training

## Implementation Architecture

### Strategy Interface
All strategies inherit from the base Strategy class and implement:
- `step()`: Core communication and optimization logic
- `_init_node()`: Node-specific initialization
- Configuration extraction via LogModule mixin

### Communication Abstraction
Strategies use communicate.py for hardware-agnostic operations:
- `all_reduce()`: Gradient/model averaging across nodes
- `broadcast()`: Parameter distribution from master node
- `all_gather()`: Metric collection and debugging

### Optimizer Integration
Strategies manage PyTorch optimizers through OptimSpec:
- Device-aware optimizer creation
- Learning rate scheduling integration
- Gradient clipping and normalization

## Usage Patterns

### Basic Data Parallel Training
```python
from exogym.strategy import SimpleReduceStrategy
strategy = SimpleReduceStrategy(optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001))
```

### Low-Communication Training
```python
from exogym.strategy import DiLoCoStrategy
strategy = DiLoCoStrategy(H=100, optim_spec=OptimSpec(torch.optim.AdamW))
```

### Federated Learning
```python
from exogym.strategy import FedAvgStrategy
strategy = FedAvgStrategy(local_epochs=5, optim_spec=OptimSpec(torch.optim.SGD))
```

### Research and Experimentation
```python
from exogym.strategy import DeMoStrategy
strategy = DeMoStrategy(alpha=0.1, optim_spec=OptimSpec(torch.optim.AdamW))
```

This module enables researchers and practitioners to experiment with cutting-edge
distributed training algorithms while maintaining compatibility with existing
PyTorch training workflows.
"""

from .strategy import Strategy
from .diloco import DiLoCoStrategy
from .optim import OptimSpec
from .sparta import SPARTAStrategy
from .federated_averaging import FedAvgStrategy
from .communicate_optimize_strategy import CommunicateOptimizeStrategy

# from .sparta_diloco import SPARTADiLoCoStrategy
from .demo import DeMoStrategy

__all__ = [
    "Strategy",
    "DiLoCoStrategy",
    "OptimSpec",
    "SPARTAStrategy",
    "FedAvgStrategy",
    "CommunicateOptimizeStrategy",
    "SPARTADiLoCoStrategy",
    "DeMoStrategy",
]
