# ExoGym Strategies API Reference

This document provides comprehensive API documentation for all distributed training strategies available in ExoGym.

## Strategy Base Class

All strategies inherit from the base `Strategy` class which defines the core interface.

```python
from exogym.strategy import Strategy
```

### Core Interface

Every strategy implements these key methods:

- **`step()`**: Called after each training step to perform communication and optimization
- **`zero_grad()`**: Clears gradients, delegates to underlying optimizer
- **`_init_node(model, rank, num_nodes)`**: Initialize strategy for specific node

### Common Parameters

All strategies accept these common parameters in their constructor:

- **`optim_spec`** (`OptimSpec`): Optimizer specification (see [OptimSpec](#optimspec))
- **`lr_scheduler`** (`str`, optional): Learning rate scheduler type (`"lambda_cosine"`, `"constant"`)
- **`lr_scheduler_kwargs`** (`dict`, optional): Additional arguments for LR scheduler
- **`max_norm`** (`float`, optional): Gradient clipping threshold

## Core Strategies

### SimpleReduceStrategy

Basic gradient averaging strategy equivalent to PyTorch DistributedDataParallel.

```python
from exogym.strategy import SimpleReduceStrategy
from exogym.strategy.optim import OptimSpec
import torch

strategy = SimpleReduceStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001, weight_decay=0.01)
)
```

**Communication Pattern**: Averages gradients across all nodes after every training step.

**Use Cases**:
- High-bandwidth networks
- Small models where communication cost is minimal
- Baseline comparisons

**Performance**: 
- ✅ Fastest convergence
- ❌ Highest communication overhead

---

### DiLoCoStrategy

Distributed Low-Communication training with periodic model averaging.

```python
from exogym.strategy import DiLoCoStrategy

strategy = DiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4),
    outer_optim="sgd",        # Outer optimizer type
    H=100,                    # Communication interval
    inner_lr=1e-4,           # Inner optimizer learning rate
    outer_lr=0.7,            # Outer optimizer learning rate
    nesterov=True            # Use Nesterov momentum for outer optimizer
)
```

**Algorithm Overview**:
1. **Inner Loop**: Each node trains locally for `H` steps
2. **Communication**: Average models across nodes every `H` steps  
3. **Outer Loop**: Apply momentum-based outer optimizer (rank 0 only)
4. **Broadcast**: Send updated model to all nodes

**Key Parameters**:
- **`H`** (`int`): Communication interval (higher = less communication)
- **`outer_optim`** (`str`): Outer optimizer (`"sgd"`, `"sgd_normalized"`)
- **`inner_lr`** (`float`): Learning rate for inner (local) optimizer
- **`outer_lr`** (`float`): Learning rate for outer (global) optimizer
- **`nesterov`** (`bool`): Use Nesterov momentum in outer optimizer

**Communication Reduction**: 10-100x less communication than SimpleReduce

**Use Cases**:
- Bandwidth-constrained environments
- Large language models
- Multi-datacenter training

---

### SPARTAStrategy

Sparse communication training with gradient sparsification.

```python
from exogym.strategy import SPARTAStrategy

strategy = SPARTAStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001),
    p_sparta=0.01,           # Sparsification ratio (1% of parameters)
    sparsification_mode="random"  # or "topk", "blockwise"
)
```

**Algorithm**: Communicates only a small fraction of model parameters each step.

**Key Parameters**:
- **`p_sparta`** (`float`): Fraction of parameters to communicate (0.001-0.1)
- **`sparsification_mode`** (`str`): How to select parameters (`"random"`, `"topk"`, `"blockwise"`)

**Communication Reduction**: 100-1000x less bandwidth usage

**Use Cases**:
- Extremely bandwidth-limited scenarios
- Edge device training
- Federated learning with mobile clients

---

### FedAvgStrategy

Federated learning with client-server architecture.

```python
from exogym.strategy import FedAvgStrategy

strategy = FedAvgStrategy(
    optim_spec=OptimSpec(torch.optim.SGD, lr=0.01),
    local_epochs=5,          # Local training epochs before communication
    island_size=4,           # Nodes per federation island
    client_sampling=0.1      # Fraction of clients to sample each round
)
```

**Algorithm**: Simulates federated learning with periodic model aggregation.

**Key Parameters**:
- **`local_epochs`** (`int`): Training epochs before communication
- **`island_size`** (`int`): Number of nodes in each federation island
- **`client_sampling`** (`float`): Fraction of clients participating each round

**Use Cases**:
- Privacy-preserving training
- Cross-device federated learning
- Heterogeneous data distributions

---

### DeMoStrategy

Research implementation of Decoupled Momentum optimization.

```python
from exogym.strategy import DeMoStrategy

strategy = DeMoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001),
    compression_decay=0.999,    # Residual accumulation rate
    compression_topk=32,        # DCT components to transmit
    compression_chunk=64        # DCT chunk size
)
```

**Algorithm**: Advanced gradient compression using DCT (Discrete Cosine Transform).

**Key Parameters**:
- **`compression_decay`** (`float`): Rate for accumulating residual gradients
- **`compression_topk`** (`int`): Number of frequency components to transmit
- **`compression_chunk`** (`int`): Size of DCT chunks for compression

**Use Cases**:
- Research and experimentation
- Advanced compression techniques
- Frequency-domain gradient analysis

## OptimSpec

Helper class for specifying optimizers in a device-agnostic way.

```python
from exogym.strategy.optim import OptimSpec
import torch

# Basic optimizer spec
optim_spec = OptimSpec(torch.optim.AdamW, lr=0.001, weight_decay=0.01)

# Advanced optimizer with scheduler
optim_spec = OptimSpec(
    torch.optim.AdamW,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8
)
```

**Constructor**:
```python
OptimSpec(optimizer_class, **optimizer_kwargs)
```

**Parameters**:
- **`optimizer_class`**: PyTorch optimizer class (e.g., `torch.optim.AdamW`)
- **`**optimizer_kwargs`**: Keyword arguments passed to optimizer constructor

## Strategy Selection Guide

### Communication vs. Convergence Trade-offs

| Strategy | Communication Frequency | Convergence Speed | Bandwidth Usage | Best For |
|----------|------------------------|-------------------|-----------------|----------|
| **SimpleReduce** | Every step | Fastest | Highest | Fast networks, small models |
| **DiLoCo** | Every H steps | Good | Medium | Moderate bandwidth constraints |
| **SPARTA** | Every step (sparse) | Slower | Lowest | Severe bandwidth limits |
| **FedAvg** | Every few epochs | Variable | Low | Privacy-sensitive scenarios |

### Hardware Recommendations

**Apple Silicon (MPS)**:
```python
# Optimized for unified memory architecture
strategy = DiLoCoStrategy(
    H=50,  # Lower H for MPS efficiency
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4)
)
```

**NVIDIA GPUs (CUDA)**:
```python
# Can handle higher communication frequency
strategy = DiLoCoStrategy(
    H=200,  # Higher H for bandwidth efficiency
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4)
)
```

**CPU-only**:
```python
# Use sparse communication to reduce overhead
strategy = SPARTAStrategy(
    p_sparta=0.01,
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-3)
)
```

## Advanced Usage

### Custom Strategy Development

```python
from exogym.strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
        
    def step(self):
        """Custom communication and optimization logic."""
        # Implement your distributed training algorithm
        self.communicate_gradients()  # or self.communicate_models()
        self.optimizer.step()
        super().step()  # Handle LR scheduling
        
    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)
        # Node-specific setup
```

### Learning Rate Scheduling

All strategies support advanced LR scheduling:

```python
strategy = DiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4),
    lr_scheduler="lambda_cosine",
    lr_scheduler_kwargs={
        "warmup_steps": 1000,     # Linear warmup for 1000 steps
        "cosine_anneal": True,    # Cosine annealing after warmup
        "min_lr_factor": 0.1      # Minimum LR = 10% of initial LR
    }
)
```

### Gradient Clipping

Enable gradient clipping for training stability:

```python
strategy = DiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4),
    max_norm=1.0  # Clip gradients to max norm of 1.0
)
```

## Error Handling

Strategies automatically handle common distributed training issues:

- **MPS Compatibility**: Automatic CPU fallback for unsupported operations
- **Memory Pressure**: Built-in memory management for Apple Silicon
- **Communication Failures**: Graceful degradation and retry logic

## See Also

- **[Core API](../../exogym/__init__.py)**: Architecture overview and data flow
- **[Communication](../../exogym/strategy/communicate.py)**: Low-level communication primitives
- **[Examples](../../example/)**: Complete training examples using all strategies
- **[Getting Started Guide](../getting_started.md)**: Step-by-step tutorial