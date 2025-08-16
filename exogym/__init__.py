"""
ExoGym - Distributed Machine Learning Training Framework

This package provides a comprehensive framework for distributed machine learning training
with support for various communication strategies and hardware platforms.

## Architecture Overview

ExoGym implements a distributed training system with the following key components:

1. **Trainer**: High-level training orchestrator that manages multiprocessing and model averaging
   - Handles process spawning for distributed training across multiple nodes
   - Manages model state collection and averaging across training processes
   - Provides device-agnostic training setup (CPU, CUDA, MPS)

2. **TrainNode**: Individual training node that executes the core training loop
   - Manages dataset loading, batching, and distributed sampling
   - Executes forward/backward passes and communicates with other nodes
   - Handles validation, checkpointing, and logging within each process

3. **Strategy**: Pluggable training strategies that define how nodes communicate
   - SimpleReduceStrategy: Basic gradient averaging after each step
   - DiLoCoStrategy: Distributed Low-Communication training with periodic model averaging
   - Federation strategies: Federated learning patterns

4. **Communication**: MPS-compatible distributed primitives for Apple Silicon
   - Automatic CPU fallback for operations not supported on MPS devices
   - Unified interface for distributed operations across CUDA, CPU, and MPS

## Complete Data Flow and Cross-File Calling Patterns

### Training Initialization Flow

```
User Code
    ↓
LocalTrainer.__init__(model, datasets) [trainer.py]
    ↓
LocalTrainer.fit(strategy, num_nodes, ...) [trainer.py]
    ↓
TrainingConfig creation [trainer.py]
    ↓
torch.multiprocessing.spawn(_worker, nprocs=num_nodes) [trainer.py]
    ↓
_worker(rank, config, result_queue) × N processes [trainer.py]
```

### Per-Process Training Flow

```
_worker() [trainer.py]
    ↓ 
Trainer._fit_process(rank) [trainer.py]
    ↓
LocalTrainer._build_connection() [trainer.py] 
    ↓
TrainNode.__init__(model, datasets, strategy, ...) [train_node.py]
    ↓
Strategy._init_node(model, rank, num_nodes) [strategy/]
    ↓
TrainNode.train() [train_node.py]
```

### Training Loop Data Flow

```
TrainNode.train() [train_node.py]
    ↓
Logger creation (WandbLogger/CSVLogger) [logger.py]
    ↓
while local_step < max_steps:
    ↓
    TrainNode._evaluate() [train_node.py] (every val_interval steps)
        ↓
        Logger.log_loss() [logger.py]
    ↓
    TrainNode._train_step() [train_node.py] (every step)
        ↓
        Strategy.zero_grad() [strategy/]
        ↓
        for minibatch in gradient_accumulation:
            ↓
            model.forward() + loss.backward()
        ↓
        Strategy.step() [strategy/]
            ↓
            SimpleReduceStrategy: communicate.all_reduce(gradients) [strategy/communicate.py]
            OR
            DiLoCoStrategy: communicate.all_reduce(models) every H steps [strategy/communicate.py]
        ↓
        Logger.log_train() [logger.py]
        ↓
        Logger.increment_step() [logger.py]
```

### Configuration and Logging Data Flow

```
Logger.__init__() [logger.py]
    ↓
utils.create_config(model, strategy, train_node) [utils.py]
    ↓
Strategy.__config__() [strategy/] + TrainNode.__config__() [train_node.py]
    ↓
LogModule.__config__() [utils.py]
    ↓
utils.extract_config(obj, max_depth=10) [utils.py]
    ↓
Configuration persistence:
    - WandbLogger: wandb.init(config) [logger.py]
    - CSVLogger: json.dump(config) [logger.py]
```

### Strategy Communication Patterns

#### SimpleReduceStrategy Flow
```
Strategy.step() [strategy/strategy.py]
    ↓
communicate.all_reduce(param.grad) for each parameter [strategy/communicate.py]
    ↓
mps_compatible decorator [strategy/communicate.py]
    ↓
torch.distributed.all_reduce() (CPU fallback for MPS) [strategy/communicate.py]
```

#### DiLoCoStrategy Flow  
```
Strategy.step() [strategy/diloco.py]
    ↓
Every step: inner_optimizer.step()
    ↓
Every H steps:
    ↓
    _average_models() [strategy/diloco.py]
        ↓
        communicate.all_reduce(param.data) [strategy/communicate.py]
    ↓
    Rank 0: _set_master_grad() + outer_optimizer.step() [strategy/diloco.py]
    ↓
    _broadcast_model_params() [strategy/diloco.py]
        ↓
        communicate.broadcast(param.data, src=0) [strategy/communicate.py]
```

### Model Aggregation Flow

```
Training completion in each worker [train_node.py]
    ↓
TrainNode.train() returns model.state_dict() [train_node.py]
    ↓
_worker() converts to CPU and puts in result_queue [trainer.py]
    ↓
Trainer.fit() collects from all workers [trainer.py]
    ↓
_average_model_states(collected_states) [trainer.py]
    ↓
Final model with averaged parameters returned to user
```

## Key Cross-File Dependencies

### Import Hierarchy
- **trainer.py** → train_node.py, strategy/
- **train_node.py** → strategy/, logger.py, utils.py  
- **strategy/** → utils.py, strategy/communicate.py
- **logger.py** → utils.py
- **utils.py** → (no internal dependencies)

### Communication Layer Abstraction
- **strategy/communicate.py** provides hardware-agnostic distributed operations
- **strategy/** classes use communicate.py for all distributed communication
- **MPS compatibility** handled transparently by mps_compatible decorator

### Configuration System
- **utils.LogModule** mixin enables automatic config extraction
- **utils.extract_config()** recursively serializes complex objects
- **utils.create_config()** aggregates training configuration
- **logger.py** uses configuration for experiment tracking

## Usage Pattern

```python
from exogym import LocalTrainer
from exogym.strategy import DiLoCoStrategy

trainer = LocalTrainer(model, train_dataset, val_dataset)
strategy = DiLoCoStrategy(H=100)  # Communicate every 100 steps

final_model = trainer.fit(
    num_epochs=10,
    strategy=strategy,
    num_nodes=4,
    batch_size=32
)
```

## File Organization

- trainer.py: Main training orchestration and multiprocessing management
- train_node.py: Individual node training implementation
- strategy/: Communication strategies for distributed training
- logger.py: WandB and CSV logging implementations
- utils.py: Configuration extraction and model utilities

## Hardware Support

- **CUDA**: Full support with NCCL backend for GPU communication
- **MPS (Apple Silicon)**: Supported with CPU fallback for unsupported operations
- **CPU**: Gloo backend for CPU-only distributed training
- **Multi-GPU**: Automatic device assignment across available GPUs

This framework is designed to be AI-developer friendly with extensive documentation
that explains both implementation details and the reasoning behind architectural decisions.
"""

from .train_node import TrainNode
from .trainer import Trainer, LocalTrainer

__all__ = ["TrainNode", "Trainer", "LocalTrainer"]
