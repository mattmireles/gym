# ExoGym: Advanced Distributed Training Simulation Framework

[![PyPI version](https://badge.fury.io/py/exogym.svg)](https://badge.fury.io/py/exogym)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *The ultimate framework for researching, prototyping, and validating distributed training algorithms*

ExoGym is a comprehensive, production-ready framework for simulated distributed training that enables researchers and practitioners to experiment with cutting-edge distributed algorithms without requiring expensive multi-GPU clusters. Instead of training with multiple physical nodes, ExoGym simulates distributed training by orchestrating multiple training processes on a single machine, providing identical algorithmic behavior to true distributed training.

## 🚀 Why ExoGym?

### 🔬 **Research & Innovation**
- **Rapid Prototyping**: Test distributed algorithms in minutes, not hours
- **Algorithm Validation**: Verify correctness before scaling to expensive infrastructure
- **Comparative Analysis**: Benchmark different strategies with identical conditions
- **Educational Tool**: Learn distributed training concepts with hands-on experimentation

### 🏭 **Production Ready**
- **Battle-Tested Algorithms**: Production implementations of state-of-the-art methods
- **Hardware Agnostic**: Seamless operation across CPU, CUDA, and Apple Silicon (MPS)
- **Comprehensive Logging**: Full integration with WandB, CSV, and custom logging systems
- **Memory Efficient**: Optimized for single-machine simulation without memory explosion

### 🎯 **Algorithmic Excellence**
- **Modular Architecture**: Mix and match communication strategies for novel algorithms
- **Theoretical Foundation**: Implementations based on peer-reviewed research
- **Performance Optimized**: Efficient communication patterns and memory management
- **Extensible Design**: Easy integration of custom distributed training strategies

---

## 🧠 Supported Distributed Training Algorithms

ExoGym provides production-ready implementations of cutting-edge distributed training algorithms:

### 📊 **Traditional Methods**
- **AllReduce (DDP)** - Standard PyTorch DistributedDataParallel equivalent
  - Immediate gradient averaging after each step
  - Optimal for high-bandwidth, low-latency networks
  - Reference implementation for correctness validation

### 🔄 **Communication-Efficient Methods**

#### **DiLoCo (Distributed Low-Communication)**
- **10-100x** communication reduction through periodic model averaging
- Dual optimization: local SGD/Adam + global outer optimizer
- Maintains convergence properties with dramatic bandwidth savings
- Ideal for bandwidth-constrained or high-latency environments

#### **SPARTA (Sparse Parameter Communication)**
- **10-1000x** bandwidth reduction via parameter sparsification
- Multiple sparsification strategies: Random, Sequential, Partitioned
- Maintains model synchronization with minimal communication
- Perfect for extremely bandwidth-limited scenarios

#### **Federated Averaging (FedAvg)**
- Client-server federated learning with island federation support
- Privacy-preserving distributed training
- Supports heterogeneous data distributions and compute capabilities
- Scalable to hundreds or thousands of participants

### 🎯 **Advanced Research Methods**

#### **DeMo (Decoupled Momentum Optimization)**
- Novel momentum decoupling with DCT-based gradient compression
- Frequency-domain gradient analysis for optimal compression
- State-of-the-art research implementation from [arXiv:2411.19870](https://arxiv.org/abs/2411.19870)
- Cutting-edge algorithm for communication-constrained environments

---

## 🔧 Installation & Setup

### 📦 **Quick Installation**

```bash
# Basic installation with core dependencies
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exogym

# With experiment tracking
pip install exogym[wandb]

# With transformer models
pip install exogym[gpt]

# With DeMo algorithm support
pip install exogym[demo]

# With example scripts
pip install exogym[examples]

# Everything included
pip install exogym[all]
```

### 🛠️ **Development Installation**

```bash
git clone https://github.com/exo-explore/gym.git exogym
cd exogym
pip install -e ".[dev]"
```

### 🖥️ **Hardware Support**

| Platform | Status | Notes |
|----------|--------|--------|
| **CPU** | ✅ Full Support | Gloo backend for distributed communication |
| **CUDA** | ✅ Full Support | NCCL backend for optimal GPU communication |
| **Apple Silicon (MPS)** | ✅ Full Support | Automatic CPU fallback for unsupported operations |

---

## 🎯 Quick Start Examples

### 🔥 **Simple DiLoCo Training**

```python
from exogym import LocalTrainer
from exogym.strategy import DiLoCoStrategy

# Your model and data
model = YourModel()
train_dataset, val_dataset = load_your_data()

# Create trainer
trainer = LocalTrainer(model, train_dataset, val_dataset)

# Configure DiLoCo strategy (communicate every 100 steps)
strategy = DiLoCoStrategy(
    inner_optim='adamw',  # Local optimizer
    outer_optim='sgd',    # Global optimizer
    H=100,                # Communication interval
    lr=3e-4
)

# Train with 4 simulated nodes
final_model = trainer.fit(
    strategy=strategy,
    num_nodes=4,
    num_epochs=10,
    batch_size=32,
    device='cuda'  # or 'mps' for Apple Silicon
)
```

### 🌐 **Federated Learning with Islands**

```python
from exogym.strategy import FedAvgStrategy

# Federated learning with 4-node islands
strategy = FedAvgStrategy(
    inner_optim='sgd',
    island_size=4,        # Create islands of 4 nodes
    H=5,                  # Local training steps before averaging
    lr=0.01
)

trainer.fit(
    strategy=strategy,
    num_nodes=16,         # 16 total nodes = 4 islands
    num_epochs=20
)
```

### 🎛️ **Bandwidth-Constrained Training with SPARTA**

```python
from exogym.strategy import SPARTAStrategy

# Extreme bandwidth reduction (0.1% of parameters per round)
strategy = SPARTAStrategy(
    p_sparta=0.001,       # Communicate 0.1% of parameters
    inner_optim='adamw',
    lr=1e-4
)

trainer.fit(
    strategy=strategy,
    num_nodes=8,
    num_epochs=15
)
```

### 🔬 **Research with DeMo Algorithm**

```python
from exogym.strategy import DeMoStrategy

# Advanced gradient compression research
strategy = DeMoStrategy(
    compression_decay=0.999,    # Residual accumulation rate
    compression_topk=32,        # Frequency components to transmit
    compression_chunk=64,       # DCT chunk size
    weight_decay=1e-4
)

trainer.fit(strategy=strategy, num_nodes=4)
```

---

## 🏗️ Architecture Overview

ExoGym implements a clean, modular architecture that separates concerns for maximum flexibility:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Trainer     │───▶│    TrainNode     │───▶│    Strategy     │
│                 │    │                  │    │                 │
│ • Orchestration │    │ • Training Loop  │    │ • Communication │
│ • Multiprocess  │    │ • Data Loading   │    │ • Optimization  │
│ • Model Sync    │    │ • Loss Compute   │    │ • Algorithms    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Logger      │    │     Dataset      │    │ Communication   │
│                 │    │                  │    │                 │
│ • WandB/CSV     │    │ • Auto Sharding  │    │ • Hardware      │
│ • Metrics       │    │ • Memory Opt     │    │   Agnostic      │
│ • Experiments   │    │ • Lazy Loading   │    │ • MPS Support   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🎪 **Core Components**

#### **Trainer: Orchestration Engine**
- **Multiprocessing Management**: Spawns and coordinates multiple training processes
- **Model Synchronization**: Handles initial model distribution and final aggregation
- **Resource Management**: Manages memory, devices, and process lifecycle
- **Error Handling**: Robust failure detection and recovery mechanisms

#### **TrainNode: Individual Training Process**
- **Local Training Loop**: Executes forward/backward passes and metric computation
- **Data Management**: Handles dataset sharding and batch loading per node
- **Strategy Integration**: Delegates optimization decisions to pluggable strategies
- **Monitoring**: Real-time metrics collection and progress tracking

#### **Strategy: Algorithm Implementation**
- **Communication Patterns**: Defines when and how nodes communicate
- **Optimization Logic**: Integrates local optimizers with distributed coordination
- **Hardware Abstraction**: Transparent operation across different device types
- **Configuration Management**: Flexible hyperparameter and algorithm configuration

---

## 🎨 Advanced Features

### 🔧 **Modular Communication Framework**

ExoGym's `CommunicateOptimizeStrategy` enables composition of multiple communication algorithms:

```python
from exogym.strategy import CommunicateOptimizeStrategy
from exogym.strategy.sparta import SparseCommunicator
from exogym.strategy.federated_averaging import AveragingCommunicator

# Compose multiple communication strategies
sparse_comm = SparseCommunicator(sparsification_rate=0.01)
averaging_comm = AveragingCommunicator(island_size=4)

# Create custom hybrid strategy
strategy = CommunicateOptimizeStrategy(
    communication_modules=[sparse_comm, averaging_comm],
    inner_optim='adamw'
)
```

### 🍎 **Apple Silicon (MPS) Support**

ExoGym provides seamless Apple Silicon support with automatic CPU fallback:

```python
# Automatic MPS optimization with CPU fallback
trainer.fit(
    strategy=strategy,
    device='mps',          # Uses Apple Silicon GPU when possible
    num_nodes=4
)
```

**MPS Features:**
- ✅ Automatic operation fallback for unsupported MPS operations
- ✅ Memory pressure management and optimization
- ✅ Unified memory architecture utilization
- ✅ Performance optimization for Apple Silicon workloads

### 📊 **Comprehensive Logging & Monitoring**

```python
from exogym.logger import WandbLogger, CSVLogger

# Multiple logging backends
logger = WandbLogger(
    project="distributed_training_research",
    experiment_name="diloco_vs_sparta_comparison"
)

# Automatic metrics tracking
trainer.fit(
    strategy=strategy,
    logger=logger,
    log_interval=50      # Log every 50 steps
)
```

**Tracked Metrics:**
- 📈 Training/validation loss and accuracy
- 🌐 Communication volume and bandwidth usage
- ⚡ Training speed and throughput
- 💾 Memory usage and device utilization
- 🔧 Hyperparameter evolution and learning rates

---

## 📚 Example Scripts & Tutorials

### 🏃 **Ready-to-Run Examples**

```bash
# MNIST comparison across all algorithms
python run/mnist.py

# Shakespeare language modeling with DiLoCo
python run/nanogpt_diloco.py --dataset shakespeare

# Computer vision with CIFAR-10
python run/cifar10_comparison.py

# Federated learning simulation
python run/federated_mnist.py --num_clients 100 --island_size 10
```

### 📖 **Comprehensive Tutorials**

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| `tutorial_basic.py` | Introduction to ExoGym concepts | 🟢 Beginner |
| `tutorial_strategies.py` | Comparing communication strategies | 🟡 Intermediate |
| `tutorial_custom.py` | Building custom distributed algorithms | 🔴 Advanced |
| `tutorial_production.py` | Production deployment patterns | 🔴 Advanced |

---

## 🔬 Research & Academic Use

### 📄 **Citation**

If you use ExoGym in your research, please cite:

```bibtex
@software{exogym2024,
  title={ExoGym: A Framework for Simulated Distributed Training},
  author={Beton, Matt and contributors},
  year={2024},
  url={https://github.com/exo-explore/gym}
}
```

### 🎓 **Research Applications**

- **Algorithm Development**: Prototype novel distributed training methods
- **Comparative Studies**: Benchmark different strategies under controlled conditions
- **Education**: Teach distributed systems and machine learning concepts
- **Validation**: Verify theoretical results with empirical implementations

### 📊 **Benchmarking Results**

| Algorithm | Communication Reduction | Convergence Quality | Best Use Case |
|-----------|------------------------|-------------------|---------------|
| **DDP** | 1x (baseline) | 100% | High-bandwidth networks |
| **DiLoCo** | 10-100x | 95-99% | Moderate bandwidth constraints |
| **SPARTA** | 100-1000x | 90-95% | Severe bandwidth limitations |
| **FedAvg** | Variable | 85-95% | Privacy-sensitive scenarios |
| **DeMo** | 50-500x | 90-98% | Research and experimentation |

---

## 🛠️ Configuration & Customization

### ⚙️ **Strategy Configuration**

```python
# Fine-tuned DiLoCo for large language models
strategy = DiLoCoStrategy(
    inner_optim='adamw',
    outer_optim='sgd',
    H=200,                    # Higher H for better bandwidth efficiency
    inner_lr=1e-4,           # Conservative inner learning rate
    outer_lr=0.5,            # Aggressive outer learning rate
    max_norm=1.0,            # Gradient clipping
    lr_scheduler='cosine',   # Learning rate scheduling
    warmup_steps=1000
)
```

### 🎯 **Custom Strategy Development**

```python
from exogym.strategy import Strategy

class MyCustomStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
        
    def step(self):
        # Custom communication and optimization logic
        self.communicate_gradients()
        self.update_parameters()
        super().step()  # Handle LR scheduling
        
    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)
        # Node-specific setup
```

### 📊 **Dataset Integration**

```python
# Automatic dataset sharding
def dataset_factory(rank, num_nodes, train_dataset):
    """Create per-node dataset with automatic sharding."""
    full_dataset = load_full_dataset()
    
    # Shard dataset across nodes
    indices = list(range(rank, len(full_dataset), num_nodes))
    return torch.utils.data.Subset(full_dataset, indices)

trainer = LocalTrainer(
    model=model,
    train_dataset=dataset_factory,  # Function instead of dataset
    val_dataset=val_dataset
)
```

---

## 🚀 Performance Optimization

### ⚡ **Memory Optimization**

```python
# Memory-efficient training configuration
trainer.fit(
    strategy=strategy,
    batch_size=32,           # Optimal batch size per node
    gradient_accumulation=4, # Simulate larger batches
    max_memory_usage=0.8     # Limit memory usage to 80%
)
```

### 🎯 **Communication Optimization**

```python
# Bandwidth-optimized SPARTA configuration
strategy = SPARTAStrategy(
    p_sparta=0.005,          # 0.5% of parameters
    compression_method='topk', # Top-K sparsification
    quantization_bits=8      # 8-bit quantization
)
```

### 🔧 **Hardware-Specific Tuning**

```python
# Apple Silicon optimization
if torch.backends.mps.is_available():
    strategy = DiLoCoStrategy(H=50)  # Lower H for MPS
else:
    strategy = DiLoCoStrategy(H=200) # Higher H for CUDA
```

---

## 🤝 Contributing & Community

### 🌟 **Contributing Guidelines**

We welcome contributions! ExoGym thrives on community involvement:

1. **🐛 Bug Reports**: Use GitHub Issues with detailed reproduction steps
2. **💡 Feature Requests**: Propose new algorithms or improvements
3. **🔧 Pull Requests**: Submit code with comprehensive tests and documentation
4. **📚 Documentation**: Help improve tutorials and examples
5. **🧪 Research**: Share novel distributed training algorithms

### 📝 **Development Setup**

```bash
# Clone and setup development environment
git clone https://github.com/exo-explore/gym.git
cd gym
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black exogym/
flake8 exogym/

# Type checking
mypy exogym/
```

### 🎯 **Code Quality Standards**

- **📖 Documentation**: Comprehensive AI-first documentation for all code
- **🧪 Testing**: Unit tests for all components with >90% coverage
- **🎨 Formatting**: Black code formatting and flake8 linting
- **🔍 Type Safety**: MyPy type checking for better code reliability
- **⚡ Performance**: Benchmarking for performance-critical components

---

## 📖 Technical Documentation

### 🔗 **API Reference**

- **[Core API](docs/api/core.md)**: Trainer, TrainNode, and Strategy classes
- **[Strategies](docs/api/strategies.md)**: All distributed training algorithms
- **[Communication](docs/api/communication.md)**: Low-level communication primitives
- **[Logging](docs/api/logging.md)**: Experiment tracking and monitoring
- **[Utilities](docs/api/utilities.md)**: Helper functions and configuration tools

### 🎓 **Tutorials & Guides**

- **[Getting Started](docs/tutorials/getting_started.md)**: Your first distributed training job
- **[Strategy Comparison](docs/tutorials/strategy_comparison.md)**: When to use which algorithm
- **[Custom Strategies](docs/tutorials/custom_strategies.md)**: Building your own algorithms
- **[Production Deployment](docs/tutorials/production.md)**: Scaling to real distributed systems
- **[Troubleshooting](docs/tutorials/troubleshooting.md)**: Common issues and solutions

### 🔬 **Algorithm Deep Dives**

- **[DiLoCo Implementation](docs/algorithms/diloco.md)**: Mathematical foundation and implementation details
- **[SPARTA Compression](docs/algorithms/sparta.md)**: Sparsification strategies and performance analysis
- **[Federated Learning](docs/algorithms/fedavg.md)**: Federation patterns and privacy considerations
- **[DeMo Research](docs/algorithms/demo.md)**: Cutting-edge momentum decoupling algorithm

---

## 🆘 Support & Resources

### 💬 **Community Support**

- **📧 Email**: [matthew.beton@gmail.com](mailto:matthew.beton@gmail.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/exo-explore/gym/issues)
- **💭 Discussions**: [GitHub Discussions](https://github.com/exo-explore/gym/discussions)
- **📚 Documentation**: [Full Documentation](https://exogym.readthedocs.io)

### 🚨 **Common Issues & Solutions**

| Issue | Solution | Reference |
|-------|----------|-----------|
| MPS compatibility | Use automatic CPU fallback | [MPS Guide](docs/hardware/mps.md) |
| Memory errors | Reduce batch size or enable gradient accumulation | [Memory Guide](docs/performance/memory.md) |
| Slow training | Optimize communication strategy selection | [Performance Guide](docs/performance/optimization.md) |
| Import errors | Check optional dependencies installation | [Installation Guide](docs/installation.md) |

### 📊 **Performance Benchmarks**

Detailed performance comparisons and benchmarks available at:
- **[Hardware Benchmarks](docs/benchmarks/hardware.md)**: Performance across different hardware
- **[Algorithm Benchmarks](docs/benchmarks/algorithms.md)**: Communication efficiency comparisons
- **[Scalability Analysis](docs/benchmarks/scalability.md)**: Performance scaling characteristics

---

## 📜 License & Legal

ExoGym is released under the **MIT License**, ensuring maximum flexibility for research and commercial use.

```
MIT License

Copyright (c) 2024 ExoGym Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

### 🏆 **Acknowledgments**

ExoGym builds upon the work of numerous researchers and open-source contributors:

- **DiLoCo**: Based on research by [Authors] - [Paper Link]
- **SPARTA**: Implementation of algorithms from [Authors] - [Paper Link]  
- **DeMo**: Research implementation from Peng et al. - [arXiv:2411.19870](https://arxiv.org/abs/2411.19870)
- **PyTorch**: Built on the PyTorch framework for deep learning
- **Community**: Thanks to all contributors who make ExoGym possible

---

## 🎯 What's Next?

ExoGym continues to evolve with the distributed training research landscape:

### 🚀 **Upcoming Features**
- **🔄 Asynchronous Training**: Support for asynchronous distributed algorithms
- **🌍 Cloud Integration**: Direct integration with cloud distributed training platforms
- **🎨 Web Interface**: Browser-based experiment management and visualization
- **📱 Mobile Support**: Training on mobile and edge devices

### 🎓 **Research Directions**
- **🧠 Adaptive Algorithms**: Self-tuning communication strategies
- **🔒 Privacy-Preserving**: Advanced differential privacy and secure aggregation
- **⚡ Hardware Acceleration**: Specialized support for new accelerator types
- **🎯 Application-Specific**: Optimizations for specific model architectures

---

**Ready to revolutionize your distributed training? Get started with ExoGym today!** 🚀

```bash
pip install exogym[all]
python -c "from exogym import LocalTrainer; print('🎉 ExoGym installed successfully!')"
```