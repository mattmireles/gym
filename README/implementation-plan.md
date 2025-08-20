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

- **[Core API](exogym/__init__.py)**: Trainer, TrainNode, and Strategy classes - see comprehensive architecture overview
- **[Strategies](exogym/strategy/__init__.py)**: All distributed training algorithms with usage examples
- **[Communication](exogym/strategy/communicate.py)**: Low-level communication primitives with MPS support
- **[Logging](exogym/logger.py)**: WandB and CSV experiment tracking implementations
- **[Utilities](exogym/utils.py)**: Configuration extraction and model averaging functions

### 🎓 **Tutorials & Guides**

- **[Getting Started](docs/getting_started.md)**: Your first distributed training job
- **[MNIST Example](example/mnist.py)**: Basic strategy comparison with MNIST dataset
- **[NanoGPT Example](example/nanogpt.py)**: Language modeling with DiLoCo strategy
- **[DiLoCo Scaling](example/diloco_scaling_batchsize.py)**: Batch size optimization patterns
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

### 🔬 **Algorithm Deep Dives**

- **[DiLoCo Implementation](exogym/strategy/diloco.py)**: Mathematical foundation and implementation details
- **[SPARTA Compression](exogym/strategy/sparta.py)**: Sparsification strategies and performance analysis
- **[Federated Learning](exogym/strategy/federated_averaging.py)**: Federation patterns and privacy considerations
- **[DeMo Research](exogym/strategy/demo.py)**: Cutting-edge momentum decoupling algorithm

---

## 🆘 Support & Resources

### 💬 **Community Support**

- **📧 Email**: [matthew.beton@gmail.com](mailto:matthew.beton@gmail.com)
- **🐛 Issues**: [GitHub Issues](https://github.com/exo-explore/gym/issues)
- **💭 Discussions**: [GitHub Discussions](https://github.com/exo-explore/gym/discussions)
- **📚 Documentation**: [Getting Started Guide](docs/getting_started.md) and [API Reference](docs/api/strategies.md)

### 🚨 **Common Issues & Solutions**

| Issue | Solution | Reference |
|-------|----------|-----------|
| MPS compatibility | Use automatic CPU fallback | [MPS Guide](CLAUDE.md#converting-pytorch-cuda-libraries-to-apple-metal-performance-shaders) |
| Memory errors | Reduce batch size or enable gradient accumulation | [Memory Guide](CLAUDE.md#memory-management-requires-mps-specific-approaches) |
| Slow training | Optimize communication strategy selection | [Performance Guide](CLAUDE.md#performance-optimization-follows-different-principles-than-cuda) |
| Import errors | Check optional dependencies installation | See installation section above |

### 📊 **Performance Benchmarks**

Detailed performance comparisons and benchmarks available at:
- **[Hardware Benchmarks](CLAUDE.md#hardware-specific-optimizations-matter-more-than-with-cuda)**: Performance across different Apple Silicon variants
- **[Algorithm Benchmarks](README.md#-research--academic-use)**: Communication efficiency comparisons in benchmarking table
- **[MPS Performance Analysis](CLAUDE.md#performance-expectations-based-on-production-benchmarks)**: Apple Silicon performance characteristics

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

# Multi-Mac Whisper Training Implementation Plan

## 🎯 Goal: Reliable Multi-Machine Distributed Training

Get Whisper models training across multiple Macs using ExoGym's advanced distributed strategies (DiLoCo, SPARTA, etc.) with a simple, hacker-friendly approach.

## 🏗 Architecture Overview

### Core Principle: Extend, Don't Replace

We're extending the existing ExoGym framework to work across actual machines instead of just simulating distributed training locally. The beauty is that ExoGym's strategies already use `torch.distributed` primitives - we just need to change the process spawning mechanism.

### Current State → Target State

**Current**: `LocalTrainer` uses `torch.multiprocessing.spawn()` to simulate distributed training on one machine

**Target**: `NetworkTrainer` uses `torch.distributed` with TCP initialization to coordinate training across multiple machines

## 📁 File Structure

```
whisper-fine-tuner-macos/
├── distributed/                    # New distributed training module
│   ├── __init__.py                 # Module initialization and exports
│   ├── network_trainer.py          # NetworkTrainer class extending ExoGym
│   ├── whisper_wrapper.py          # Whisper model adapter for ExoGym interface
│   ├── launcher.py                 # SSH-based multi-machine launcher
│   └── utils.py                    # Distributed training utilities
├── distributed_hosts.json          # Simple host configuration (user-created)
├── wizard.py                       # Enhanced with distributed training option
├── cli_typer.py                    # New distributed-train command
├── config.ini                      # New [distributed:*] profiles
└── gym/                            # Existing ExoGym (minimal changes)
    └── README/
        └── implementation-plan.md   # This file
```

## 🔧 Technical Implementation

### 1. Network Trainer (`distributed/network_trainer.py`)

```python
class NetworkTrainer(Trainer):
    """
    Multi-machine distributed trainer extending ExoGym's LocalTrainer.
    
    Key differences from LocalTrainer:
    - Uses torch.distributed.init_process_group() with TCP backend
    - Spawns processes across multiple machines via SSH
    - Handles network-aware error recovery and monitoring
    """
    
    def __init__(self, hosts_config, model, train_dataset, val_dataset, **kwargs):
        self.hosts_config = hosts_config  # Load from distributed_hosts.json
        self.master_addr = hosts_config['master']
        self.worker_addrs = hosts_config['workers']
        super().__init__(model, train_dataset, val_dataset, **kwargs)
    
    def _build_connection(self):
        """
        Initialize torch.distributed for multi-machine training.
        Uses TCP backend for network communication.
        """
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.port)
        
        # Initialize process group for network-based distributed training
        dist.init_process_group(
            backend="nccl" if self.device == "cuda" else "gloo",
            init_method=f"tcp://{self.master_addr}:{self.port}",
            rank=self.rank,
            world_size=self.num_nodes
        )
    
    def fit(self, **kwargs):
        """
        Override fit to launch processes across multiple machines.
        """
        # Use launcher to start processes on remote machines
        from .launcher import DistributedLauncher
        launcher = DistributedLauncher(self.hosts_config)
        return launcher.launch_training(self, **kwargs)
```

### 2. Whisper Model Wrapper (`distributed/whisper_wrapper.py`)

```python
class WhisperModelWrapper(nn.Module):
    """
    Adapter that makes Whisper models compatible with ExoGym's training interface.
    
    ExoGym expects models to:
    - Take batches and return loss directly
    - Handle device movement automatically
    - Support standard optimizer interfaces
    
    This wrapper handles Whisper's encoder-decoder architecture and mel spectrogram preprocessing.
    """
    
    def __init__(self, whisper_model, processor, device):
        super().__init__()
        self.whisper_model = whisper_model
        self.processor = processor
        self.device = device
    
    def forward(self, batch):
        """
        ExoGym interface: batch → loss
        
        Handles:
        - Audio preprocessing to mel spectrograms
        - Encoder-decoder forward pass
        - Loss computation for ASR task
        """
        audio_features, labels = batch
        
        # Whisper-specific preprocessing
        mel_features = self.processor(audio_features, sampling_rate=16000)
        
        # Forward pass through Whisper encoder-decoder
        outputs = self.whisper_model(
            input_features=mel_features,
            labels=labels
        )
        
        return outputs.loss
```

### 3. SSH Launcher (`distributed/launcher.py`)

```python
class DistributedLauncher:
    """
    Launches and coordinates training processes across multiple machines via SSH.
    
    Simple, reliable approach:
    - SSH into each worker machine
    - Start training process with appropriate rank and world_size
    - Monitor processes and collect results
    - Handle failures gracefully
    """
    
    def __init__(self, hosts_config):
        self.master_addr = hosts_config['master']
        self.worker_addrs = hosts_config['workers']
        self.ssh_user = hosts_config.get('ssh_user', 'username')
    
    def launch_training(self, trainer, **training_kwargs):
        """
        1. Validate SSH connectivity to all machines
        2. Launch worker processes via SSH
        3. Run master process locally
        4. Coordinate training and collect results
        """
        # Pre-flight checks
        self._validate_connectivity()
        self._check_remote_dependencies()
        
        # Launch workers
        worker_processes = []
        for rank, worker_addr in enumerate(self.worker_addrs, 1):
            cmd = self._build_worker_command(rank, trainer, **training_kwargs)
            process = self._ssh_launch(worker_addr, cmd)
            worker_processes.append(process)
        
        # Run master locally
        trainer.rank = 0
        trainer.num_nodes = len(self.worker_addrs) + 1
        result = trainer._fit_process(rank=0)
        
        # Wait for workers and collect results
        self._wait_for_workers(worker_processes)
        
        return result
    
    def _ssh_launch(self, host, command):
        """Launch command on remote host via SSH."""
        ssh_cmd = f"ssh {self.ssh_user}@{host} '{command}'"
        return subprocess.Popen(ssh_cmd, shell=True)
    
    def _build_worker_command(self, rank, trainer, **kwargs):
        """Build the command to run on worker machines."""
        # Serialize trainer and arguments for remote execution
        # This is the tricky part - we need to recreate the training context on remote machines
        return f"cd {os.getcwd()} && python -m distributed.worker_main --rank {rank} --config '{serialize_config(trainer, kwargs)}'"
```

### 4. Configuration Support (`distributed_hosts.json`)

```json
{
  "master": "192.168.1.100",
  "workers": [
    "192.168.1.101", 
    "192.168.1.102"
  ],
  "ssh_user": "matt",
  "ssh_key_path": "~/.ssh/id_rsa",
  "python_env": "/opt/anaconda3/envs/whisper/bin/python",
  "project_path": "/Users/matt/whisper-fine-tuner-macos"
}
```

### 5. Enhanced Wizard Integration (`wizard.py`)

```python
def check_distributed_training():
    """
    Check if distributed training is configured and offer it as an option.
    
    Beautiful progressive disclosure:
    1. Check for distributed_hosts.json
    2. Validate connectivity to configured machines
    3. Offer distributed training with estimated speedup
    """
    hosts_file = Path("distributed_hosts.json")
    if not hosts_file.exists():
        return None
    
    try:
        with open(hosts_file) as f:
            hosts_config = json.load(f)
        
        # Quick connectivity check
        available_workers = validate_workers(hosts_config['workers'])
        
        if available_workers:
            console.print(Panel(
                f"🚀 I found {len(available_workers)} other Macs configured for distributed training!\n"
                f"📊 Estimated speedup: {len(available_workers) + 1}x faster\n"
                f"🎯 Strategy: DiLoCo (minimal communication, maximum speed)",
                title="Distributed Training Available",
                border_style="blue"
            ))
            
            use_distributed = questionary.confirm(
                "Use distributed training across multiple Macs?",
                default=True,
                style=apple_style
            ).ask()
            
            if use_distributed:
                return hosts_config
    except Exception as e:
        console.print(f"[yellow]Distributed training config found but invalid: {e}[/yellow]")
    
    return None

def run_distributed_training(model, dataset, hosts_config, **training_args):
    """Execute distributed training across multiple machines."""
    from distributed import NetworkTrainer, WhisperModelWrapper
    
    # Wrap Whisper model for ExoGym compatibility
    wrapped_model = WhisperModelWrapper(model, processor, device)
    
    # Create network trainer
    trainer = NetworkTrainer(
        hosts_config=hosts_config,
        model=wrapped_model,
        train_dataset=dataset,
        val_dataset=val_dataset
    )
    
    # Choose strategy (DiLoCo for V1)
    from gym.exogym.strategy.diloco import DiLoCoStrategy
    strategy = DiLoCoStrategy(H=100, inner_optim='adamw', outer_optim='sgd')
    
    # Launch distributed training
    console.print("\n🚀 Starting distributed training across machines...")
    final_model = trainer.fit(
        strategy=strategy,
        num_nodes=len(hosts_config['workers']) + 1,
        **training_args
    )
    
    return final_model
```

## 🔄 Integration Points

### 1. CLI Command

```bash
# New distributed training command
python cli_typer.py distributed-train medium-lora-data3 --hosts distributed_hosts.json

# Wizard with distributed option
python wizard.py
```

### 2. Config Profiles

```ini
[distributed:diloco-medium]
base_profile = medium-lora-data3
strategy = diloco
communication_interval = 100
hosts_config = distributed_hosts.json
```

## 🧪 Testing Strategy

### Phase 1: Local Testing
1. Test NetworkTrainer with localhost addresses (simulate multi-machine)
2. Validate Whisper model wrapper with different model sizes
3. Test SSH launcher with local SSH connections

### Phase 2: Multi-Machine Testing
1. Test with 2 Macs on same network
2. Test failure scenarios (network disconnection, worker failure)
3. Performance benchmarking vs single-machine training

### Phase 3: Integration Testing
1. Test wizard integration with distributed options
2. Test CLI commands and configuration
3. End-to-end training runs with real datasets

## 📈 Success Metrics

### V1 Success Criteria
- [ ] Can train Whisper models across 2+ Macs using DiLoCo
- [ ] Uses standard torch.distributed (no custom networking)
- [ ] Simple JSON config for hosts (no auto-discovery)
- [ ] Integrates with existing wizard and CLI
- [ ] Handles common failure cases gracefully
- [ ] Performance improvement scales with number of machines

### Performance Targets
- **2 Macs**: 1.8x speedup (accounting for communication overhead)
- **3 Macs**: 2.5x speedup
- **4 Macs**: 3.2x speedup

## ⚠️ Risk Mitigation

### Technical Risks
1. **Network failures**: Implement retry logic and graceful degradation
2. **SSH connectivity**: Clear error messages and setup documentation
3. **Model serialization**: Careful handling of large model state across network
4. **ExoGym integration**: Thorough testing of strategy compatibility

### User Experience Risks
1. **Complex setup**: Provide clear documentation and helper scripts
2. **Debugging difficulty**: Comprehensive logging and error reporting
3. **Version mismatches**: Environment validation across machines

## 🗓 Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Create distributed module structure
- [ ] Implement NetworkTrainer basic functionality
- [ ] Create Whisper model wrapper
- [ ] Build basic SSH launcher

### Week 2: Integration & Testing
- [ ] Integrate with wizard and CLI
- [ ] Add configuration support
- [ ] Test across multiple machines
- [ ] Documentation and polish

## 🚀 The 'Steve Jobs' Roadmap (V2+)

Once the robust V1 foundation is in place, we will iteratively build towards the effortless, magical user experience we initially envisioned. The goal is to make distributed training not just accessible, but delightful.

### The Seamless Experience
The end-user experience should feel like this:
1.  **Discovery**: The wizard automatically detects other Macs on the network. *"I found 2 other Macs. Want to use them for 3x faster training?"*
2.  **Setup**: One-click confirmation handles all SSH configuration and environment validation seamlessly. *"Setting up secure connections... ✓ All machines ready!"*
3.  **Training**: A beautiful, real-time dashboard visualizes progress, performance metrics, and health across all machines.
4.  **Intelligence**: The system intelligently selects the best distributed strategy (DiLoCo, SPARTA, etc.) based on the network conditions and model architecture.
5.  **Resilience**: The training run is resilient to network hiccups or machine failures, automatically recovering where possible.

### V2+ Feature Breakdown
- **Auto-Discovery & Zero-Conf Setup**: Use mDNS/Bonjour to find other Macs and automate the entire setup process.
- **Real-Time Web Dashboard**: A rich UI to monitor training progress, system utilization (CPU/GPU/Network), and model performance metrics across the cluster.
- **Multi-Strategy Intelligence**: Full integration of SPARTA, FedAvg, and other ExoGym strategies, with automated selection for the user's scenario.
- **Advanced Fault Tolerance**: Automatic snapshotting and recovery, allowing training to resume even if a worker node drops.
- **Dynamic Load Balancing**: Intelligently distribute workloads based on each machine's real-time capabilities and availability.

## 📚 Documentation Requirements

### User Documentation
1. **Setup Guide**: SSH configuration, dependencies, network setup
2. **Usage Guide**: CLI commands, configuration options, troubleshooting
3. **Performance Guide**: Optimal configurations for different scenarios

### Developer Documentation
1. **Architecture Overview**: System design and component interaction
2. **API Reference**: NetworkTrainer, launcher, and wrapper interfaces
3. **Extension Guide**: Adding new strategies and customizations

---

## 🎯 The Bottom Line

This plan delivers a working, reliable multi-machine distributed training system for Whisper models. It follows the "hacker way" philosophy:

- **Simple**: Uses standard tools (torch.distributed, SSH)
- **Reliable**: Built on proven technologies
- **Extensible**: Clean architecture for future enhancements
- **Practical**: Solves the real problem without over-engineering

The goal is to get you training across your Macs in 2 weeks, not 2 months.