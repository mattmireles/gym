# Troubleshooting ExoGym

This guide helps you solve common issues when using ExoGym for distributed training.

## Installation Issues

### "ModuleNotFoundError: No module named 'exogym'"

**Problem**: ExoGym not properly installed.

**Solution**:
```bash
# Try installing from test PyPI first
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exogym

# Verify installation
python -c "from exogym import LocalTrainer; print('✅ ExoGym installed')"
```

### "ImportError: No module named 'torch'"

**Problem**: PyTorch not installed or wrong version.

**Solution**:
```bash
# Install PyTorch for your platform
pip install torch torchvision

# For Apple Silicon
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch works
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## Apple Silicon (MPS) Issues

### "MPS backend not available"

**Problem**: PyTorch not built with MPS support or running under Rosetta.

**Diagnosis**:
```python
import platform
import torch

print(f"Platform: {platform.platform()}")  # Must show 'arm64', not 'x86_64'
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

**Solutions**:
1. **Wrong Python architecture**: You're running x86_64 Python under Rosetta
   ```bash
   # Check Python architecture
   python -c "import platform; print(platform.platform())"
   
   # If shows x86_64, reinstall Python/Conda for ARM64
   # Download native ARM64 Python from python.org
   ```

2. **Old PyTorch version**: Update to PyTorch 1.12+
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **macOS too old**: Requires macOS 12.3+
   ```bash
   sw_vers  # Check macOS version
   ```

### "NotImplementedError: The operator 'X' is not currently implemented for the MPS device"

**Problem**: Operation not supported on MPS, needs CPU fallback.

**Quick Fix**:
```python
# Enable automatic CPU fallback (debugging only)
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

**Proper Solution**: Update your code to handle unsupported operations:
```python
def mps_safe_operation(tensor, operation, *args):
    try:
        return operation(tensor, *args)
    except NotImplementedError:
        result = operation(tensor.cpu(), *args)
        return result.to(tensor.device)
```

### "MPS training suddenly becomes very slow"

**Problem**: Memory pressure causing disk swapping.

**Solution**:
```python
# Set memory limit (70% of system RAM)
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

# Monitor memory usage
import torch
print(f"MPS memory: {torch.mps.current_allocated_memory() / 1e9:.1f} GB")

# Clear MPS cache if needed
torch.mps.empty_cache()
```

## Training Issues

### "Training doesn't start / hangs at initialization"

**Problem**: Multiprocessing issues or device conflicts.

**Solutions**:
1. **Reduce number of nodes**:
   ```python
   # Start with fewer nodes
   trainer.fit(num_nodes=2)  # Instead of 4 or 8
   ```

2. **Check device availability**:
   ```python
   # Verify device before training
   device = "mps" if torch.backends.mps.is_available() else "cpu"
   print(f"Using device: {device}")
   ```

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### "CUDA out of memory"

**Problem**: Model or batch size too large for GPU memory.

**Solutions**:
1. **Reduce batch size**:
   ```python
   trainer.fit(batch_size=16)  # Instead of 64
   ```

2. **Use gradient accumulation**:
   ```python
   trainer.fit(
       batch_size=16,
       minibatch_size=4,  # Accumulate 4 mini-batches
       gradient_accumulation=4
   )
   ```

3. **Clear CUDA cache**:
   ```python
   torch.cuda.empty_cache()
   ```

### "Loss is NaN or training diverges"

**Problem**: Learning rate too high or gradient explosion.

**Solutions**:
1. **Reduce learning rate**:
   ```python
   strategy = DiLoCoStrategy(
       optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-5)  # Lower LR
   )
   ```

2. **Enable gradient clipping**:
   ```python
   strategy = DiLoCoStrategy(
       optim_spec=OptimSpec(torch.optim.AdamW, lr=1e-4),
       max_norm=1.0  # Clip gradients
   )
   ```

3. **Check data normalization**:
   ```python
   # Ensure proper data preprocessing
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))  # Proper normalization
   ])
   ```

## Strategy-Specific Issues

### DiLoCo Strategy Problems

**"DiLoCo converges slower than expected"**:
- **Reduce H**: Lower communication interval for faster convergence
  ```python
  strategy = DiLoCoStrategy(H=50)  # Instead of H=200
  ```
- **Tune outer LR**: Increase outer learning rate
  ```python
  strategy = DiLoCoStrategy(outer_lr=1.0)  # Instead of 0.7
  ```

### SPARTA Strategy Problems

**"SPARTA doesn't reduce communication as expected"**:
- **Check sparsification ratio**: Ensure p_sparta is small enough
  ```python
  strategy = SPARTAStrategy(p_sparta=0.001)  # 0.1% of parameters
  ```

### FedAvg Strategy Problems

**"FedAvg doesn't converge"**:
- **Increase local epochs**: More local training before communication
  ```python
  strategy = FedAvgStrategy(local_epochs=10)  # Instead of 5
  ```

## Performance Issues

### "Training is slower than expected"

**Diagnosis**:
```python
import time
import torch

# Profile device performance
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)

start = time.time()
y = torch.mm(x, x)
torch.mps.synchronize() if device.type == "mps" else None
end = time.time()

print(f"Matrix multiply took {end-start:.3f}s on {device}")
```

**Solutions**:
1. **Optimize batch size** for your hardware:
   ```python
   # Start small and increase until memory full
   for bs in [16, 32, 64, 128]:
       try:
           trainer.fit(batch_size=bs, num_epochs=1)
           print(f"Batch size {bs}: ✅")
       except RuntimeError as e:
           print(f"Batch size {bs}: ❌ {e}")
           break
   ```

2. **Reduce communication frequency**:
   ```python
   # For DiLoCo, increase H
   strategy = DiLoCoStrategy(H=100)  # Less frequent communication
   ```

3. **Use smaller models** for testing:
   ```python
   # Simplify model architecture for debugging
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(784, 128),
               nn.ReLU(),
               nn.Linear(128, 10)
           )
   ```

### "Memory usage keeps growing"

**Problem**: Memory leak or improper cleanup.

**Solutions**:
1. **Clear caches periodically**:
   ```python
   if device.type == "mps":
       torch.mps.empty_cache()
   elif device.type == "cuda":
       torch.cuda.empty_cache()
   ```

2. **Use context managers**:
   ```python
   with torch.no_grad():
       # Validation code here
       predictions = model(validation_batch)
   ```

## Communication Issues

### "Distributed training fails to start"

**Problem**: Process communication setup issues.

**Solutions**:
1. **Check multiprocessing method**:
   ```python
   import torch.multiprocessing as mp
   mp.set_start_method('spawn', force=True)  # Force spawn method
   ```

2. **Reduce number of nodes**:
   ```python
   trainer.fit(num_nodes=2)  # Start with 2 nodes
   ```

3. **Check system resources**:
   ```bash
   # Monitor system resources
   top -o cpu  # Check CPU usage
   vm_stat     # Check memory pressure (macOS)
   ```

## Data Issues

### "DataLoader errors or slow data loading"

**Problem**: Dataset loading or preprocessing issues.

**Solutions**:
1. **Reduce num_workers**:
   ```python
   # Reduce DataLoader workers
   trainer.fit(num_workers=0)  # Single-threaded loading
   ```

2. **Check dataset size**:
   ```python
   print(f"Train dataset size: {len(train_dataset)}")
   print(f"Validation dataset size: {len(val_dataset)}")
   ```

3. **Simplify transforms**:
   ```python
   # Use minimal transforms for debugging
   simple_transform = transforms.Compose([
       transforms.ToTensor()
   ])
   ```

## Environment Debugging

### Complete Environment Check

Run this diagnostic script to check your environment:

```python
import torch
import platform
import sys
import os

def diagnostic_check():
    print("=== ExoGym Environment Diagnostic ===\n")
    
    # Platform info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # Device availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Memory info
    if torch.backends.mps.is_available():
        print(f"MPS memory: {torch.mps.current_allocated_memory() / 1e9:.1f} GB")
    if torch.cuda.is_available():
        print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    # Environment variables
    relevant_vars = ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_HIGH_WATERMARK_RATIO']
    for var in relevant_vars:
        if var in os.environ:
            print(f"{var}: {os.environ[var]}")
    
    # Test basic operations
    try:
        from exogym import LocalTrainer
        from exogym.strategy import DiLoCoStrategy
        print("✅ ExoGym imports successful")
    except Exception as e:
        print(f"❌ ExoGym import failed: {e}")

if __name__ == "__main__":
    diagnostic_check()
```

## Getting Help

If you're still experiencing issues:

1. **Check the source code**: Many questions can be answered by reading:
   - [exogym/__init__.py](../exogym/__init__.py) - Architecture overview
   - [exogym/strategy/__init__.py](../exogym/strategy/__init__.py) - Strategy documentation

2. **Run diagnostic script** above to gather environment info

3. **Try minimal examples**: Start with the [getting started guide](getting_started.md)

4. **Check Apple Silicon guide**: For MPS-specific issues, see [CLAUDE.md](../CLAUDE.md)

5. **Create an issue**: If all else fails, create a GitHub issue with:
   - Output of diagnostic script
   - Minimal code example that reproduces the issue
   - Full error traceback

Remember: Most issues are environment-related (especially on Apple Silicon) or due to resource constraints. Start simple and gradually increase complexity.