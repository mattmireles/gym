# Getting Started with ExoGym

Welcome to ExoGym! This guide will help you run your first distributed training experiment in minutes.

## Quick Setup

### 1. Install ExoGym

```bash
# Basic installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exogym

# Or with all features
pip install exogym[all]
```

### 2. Verify Installation

```python
from exogym import LocalTrainer
from exogym.strategy import DiLoCoStrategy
print("🎉 ExoGym installed successfully!")
```

## Your First Distributed Training Job

Let's train a simple model using ExoGym's DiLoCo strategy. This example shows the complete flow from model definition to distributed training.

### Complete Example

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from exogym import LocalTrainer
from exogym.strategy import DiLoCoStrategy
from exogym.strategy.optim import OptimSpec

# 1. Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# 2. Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# 3. Create model and trainer
model = SimpleModel()
trainer = LocalTrainer(model, train_dataset, val_dataset)

# 4. Configure distributed strategy
strategy = DiLoCoStrategy(
    optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001),
    H=50,  # Communicate every 50 steps
)

# 5. Automatic device selection
device = (
    "mps" if torch.backends.mps.is_available()    # Apple Silicon
    else "cuda" if torch.cuda.is_available()      # NVIDIA GPU
    else "cpu"                                    # CPU fallback
)

# 6. Run distributed training
print("🚀 Starting distributed training...")
final_model = trainer.fit(
    num_epochs=3,
    strategy=strategy,
    num_nodes=4,          # Simulate 4 nodes
    device=device,
    batch_size=64,
    val_interval=100      # Validate every 100 steps
)

print("✅ Training completed!")
```

### Save and Run

1. Save this code as `my_first_exogym.py`
2. Run it: `python my_first_exogym.py`
3. Watch ExoGym simulate 4-node distributed training on your single machine!

## Understanding the Output

When you run the example, you'll see:

```
🚀 Starting distributed training...
[INFO] Starting distributed training with 4 nodes
[Node 0] Epoch 1/3, Step 100, Loss: 0.245, Val Acc: 92.1%
[Node 1] Epoch 1/3, Step 100, Loss: 0.251, Val Acc: 91.8%
[Node 2] Epoch 1/3, Step 100, Loss: 0.238, Val Acc: 92.4%
[Node 3] Epoch 1/3, Step 100, Loss: 0.243, Val Acc: 92.0%
[Communication] Averaging models across nodes (H=50 reached)
✅ Training completed!
```

## Key Concepts

### Strategies
ExoGym supports multiple distributed training strategies:

- **DiLoCoStrategy**: Communication-efficient, updates models every H steps
- **SimpleReduceStrategy**: Traditional gradient averaging (like PyTorch DDP)
- **SPARTAStrategy**: Sparse communication for bandwidth-constrained environments

### Communication Interval (H)
The `H` parameter controls how often nodes communicate:
- Small H (e.g., H=1): More communication, faster convergence, higher bandwidth
- Large H (e.g., H=100): Less communication, slower convergence, lower bandwidth

### Device Support
ExoGym automatically works on:
- **Apple Silicon (MPS)**: Optimized for M1/M2/M3 chips
- **NVIDIA GPUs (CUDA)**: Full GPU acceleration
- **CPU**: Fallback for any system

## Next Steps

### Try Different Strategies

```python
# Compare communication strategies
strategies = {
    "diloco": DiLoCoStrategy(optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001), H=50),
    "simple": SimpleReduceStrategy(optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001)),
    "sparta": SPARTAStrategy(optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001), p_sparta=0.01)
}

for name, strategy in strategies.items():
    print(f"\n=== Training with {name} ===")
    model = SimpleModel()  # Fresh model for each strategy
    trainer = LocalTrainer(model, train_dataset, val_dataset)
    trainer.fit(strategy=strategy, num_nodes=4, num_epochs=2, device=device)
```

### Explore Examples

- **[MNIST Comparison](../example/mnist.py)**: Comprehensive strategy comparison
- **[NanoGPT](../example/nanogpt.py)**: Language model training with DiLoCo
- **[Scaling Study](../example/diloco_scaling_batchsize.py)**: Batch size optimization

### Hardware Optimization

For Apple Silicon users, see our [MPS optimization guide](../CLAUDE.md#converting-pytorch-cuda-libraries-to-apple-metal-performance-shaders) for best practices.

## Need Help?

- **Source Code**: Check [exogym/__init__.py](../exogym/__init__.py) for architecture overview
- **Strategies**: See [exogym/strategy/__init__.py](../exogym/strategy/__init__.py) for all available strategies
- **Troubleshooting**: Common issues in [docs/troubleshooting.md](troubleshooting.md)

Happy distributed training! 🚀