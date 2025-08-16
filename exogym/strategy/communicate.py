"""
ExoGym Communication - MPS-Compatible Distributed Operations

This module provides hardware-agnostic wrappers around PyTorch distributed
communication primitives. The key innovation is automatic CPU fallback for
Apple Silicon MPS (Metal Performance Shaders) devices, which don't support
distributed operations natively.

## MPS Compatibility Challenge

Apple Silicon GPUs (M1, M2, M3+) use Metal Performance Shaders (MPS) for GPU
acceleration. However, PyTorch's distributed communication library was designed
for CUDA and doesn't support MPS tensors directly. This creates a problem for
distributed training on Apple hardware.

## Solution: Transparent CPU Fallback

The mps_compatible decorator automatically handles device conversion:

1. **Detection**: Check if any tensors are on MPS devices
2. **CPU Transfer**: Move MPS tensors to CPU before distributed operation
3. **Communication**: Perform standard distributed operation on CPU
4. **GPU Return**: Copy results back to MPS devices

This approach maintains the distributed training semantics while enabling
Apple Silicon support without code changes in higher-level components.

## Performance Implications

### CPU Fallback Overhead
- **Data Transfer Cost**: GPU→CPU→GPU transfers add latency
- **Bandwidth**: Limited by PCIe bandwidth, not Metal memory bandwidth  
- **Synchronization**: Forces GPU-CPU sync points

### When It's Worth It
- **Model Averaging**: Large models benefit from GPU acceleration despite transfer costs
- **Development**: Enables distributed training development on Apple hardware
- **Mixed Fleets**: Allows CUDA and MPS nodes in same training job

## Implementation Details

### Decorator Pattern
The mps_compatible decorator wraps PyTorch distributed functions and:
- Preserves original function signatures and behavior
- Handles both single tensor and tensor list operations
- Maintains in-place semantics by copying data back to original tensors

### Special Case: all_gather
all_gather takes both tensor_list and tensor arguments, requiring special handling
to detect MPS devices across both parameter types and manage bidirectional
data movement correctly.

### Memory Management
- Uses .data attribute to access underlying storage
- Performs in-place copy operations to preserve Python object identity
- No additional memory allocation beyond temporary CPU copies

## API Surface

### Supported Operations
- **broadcast**: One-to-many parameter distribution
- **all_reduce**: Sum/average across all nodes  
- **all_gather**: Collect tensors from all nodes

### Unsupported Operations (Commented Out)
- reduce_scatter, reduce, gather: Not implemented due to limited use in current strategies

## Usage Pattern

```python
from .communicate import all_reduce, broadcast

# These work transparently across CUDA, MPS, and CPU
all_reduce(gradients)  # Averages gradients across nodes
broadcast(model_params, src=0)  # Distributes from rank 0
```

## Hardware Support Matrix

| Device | Distributed | Performance | Status |
|--------|-------------|------------|--------|
| CUDA   | Native      | Optimal    | ✅ Full |
| CPU    | Native      | Good       | ✅ Full |  
| MPS    | CPU Fallback| Reduced    | ✅ Compatible |

## Called by:
- Strategy classes (DiLoCoStrategy, SimpleReduceStrategy) for distributed communication
- TrainNode for initial parameter broadcasting

## Calls:
- torch.distributed functions for actual communication operations
- Tensor device detection and data movement operations

## Future Improvements

Apple may add native MPS support to PyTorch distributed in future releases.
When available, this compatibility layer can be bypassed for optimal performance
while maintaining the same API surface.
"""

import torch.distributed as dist


def mps_compatible(func):
    # Wrapper for all_gather which handles tensor_list and tensor
    def all_gather_wrapper(tensor_list, tensor, *args, **kwargs):
        # Check if either is on MPS
        is_tensor_mps = hasattr(tensor, "device") and tensor.device.type == "mps"
        is_list_mps = any(
            hasattr(t, "device") and t.device.type == "mps" for t in tensor_list
        )

        if is_tensor_mps or is_list_mps:
            # Convert tensor to CPU if needed
            if is_tensor_mps:
                cpu_tensor = tensor.data.to("cpu")
            else:
                cpu_tensor = tensor

            # Convert tensor_list to CPU if needed
            cpu_tensor_list = []
            for t in tensor_list:
                if hasattr(t, "device") and t.device.type == "mps":
                    cpu_tensor_list.append(t.data.to("cpu"))
                else:
                    cpu_tensor_list.append(t)

            # Call function with CPU tensors
            result = func(cpu_tensor_list, cpu_tensor, *args, **kwargs)

            # Copy data back to original devices
            if is_tensor_mps:
                tensor.data.copy_(cpu_tensor.to("mps"))

            for i, t in enumerate(tensor_list):
                if hasattr(t, "device") and t.device.type == "mps":
                    t.data.copy_(cpu_tensor_list[i].to("mps"))

            return result
        else:
            return func(tensor_list, tensor, *args, **kwargs)

    # Wrapper for all other functions that handle a single tensor
    def standard_wrapper(tensor, *args, **kwargs):
        if hasattr(tensor, "device") and tensor.device.type == "mps":
            # Move the tensor to CPU
            cpu_tensor = tensor.data.to("cpu")
            # Call the function on CPU
            result = func(cpu_tensor, *args, **kwargs)
            # Copy the result back to mps
            tensor.data.copy_(cpu_tensor.to("mps"))
            return result
        else:
            return func(tensor, *args, **kwargs)

    # Return the appropriate wrapper based on function name
    if func.__name__ == "all_gather":
        return all_gather_wrapper
    else:
        return standard_wrapper


@mps_compatible
def broadcast(tensor, src=0):
    """
    Broadcast tensor from source rank to all other ranks with MPS compatibility.
    
    Distributes a tensor from the specified source rank to all other ranks in the
    distributed training group. Essential for parameter synchronization and
    initialization in distributed training.
    
    ## MPS Compatibility Handling
    
    When MPS tensors are detected:
    1. Tensor moved from MPS to CPU before broadcast operation
    2. Standard PyTorch distributed broadcast performed on CPU
    3. Result copied back to MPS device maintaining original tensor identity
    
    For non-MPS tensors (CUDA, CPU), operates as standard PyTorch broadcast.
    
    ## Usage in ExoGym
    
    - **Parameter Initialization**: Broadcast initial model parameters from rank 0
    - **DiLoCo Strategy**: Broadcast updated master model after outer optimization
    - **Configuration Sync**: Distribute configuration tensors across ranks
    
    Args:
        tensor: Tensor to broadcast (modified in-place)
        src: Source rank for broadcast operation (default: 0)
        
    Returns:
        None (tensor modified in-place)
        
    Called by:
        DiLoCoStrategy._broadcast_model_params() for model synchronization
        TrainNode.__init__() for initial parameter broadcasting
        
    Calls:
        torch.distributed.broadcast() for actual communication
    """
    return dist.broadcast(tensor, src=src)


@mps_compatible
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    Reduce tensor across all ranks with configurable operation and MPS compatibility.
    
    Performs collective reduction operation (sum, average, max, etc.) across all
    distributed training ranks. Most commonly used for gradient averaging in
    distributed training.
    
    ## Reduction Operations
    
    - **SUM** (default): Sum values across all ranks
    - **AVERAGE**: Compute average across all ranks  
    - **MAX**: Find maximum value across all ranks
    - **MIN**: Find minimum value across all ranks
    
    ## MPS Compatibility Handling
    
    When MPS tensors are detected:
    1. Tensor moved from MPS to CPU before reduction operation
    2. Standard PyTorch distributed all_reduce performed on CPU
    3. Result copied back to MPS device maintaining original tensor identity
    
    For non-MPS tensors (CUDA, CPU), operates as standard PyTorch all_reduce.
    
    ## Usage in ExoGym
    
    - **Gradient Averaging**: Sum gradients across ranks in SimpleReduceStrategy
    - **Model Averaging**: Sum model parameters across ranks in DiLoCo
    - **Metric Aggregation**: Combine validation metrics from all ranks
    
    ## Performance Considerations
    
    - **CUDA**: Optimized NCCL backend for maximum bandwidth
    - **MPS**: CPU fallback introduces transfer overhead but enables compatibility
    - **CPU**: Gloo backend for CPU-only distributed training
    
    Args:
        tensor: Tensor to reduce (modified in-place with result)
        op: Reduction operation (default: SUM)
        
    Returns:
        None (tensor modified in-place)
        
    Called by:
        SimpleReduceStrategy.step() for gradient averaging
        DiLoCoStrategy._average_models() for model parameter averaging
        
    Calls:
        torch.distributed.all_reduce() for actual communication
    """
    return dist.all_reduce(tensor, op=op)


@mps_compatible
def all_gather(tensor_list, tensor, group=None, async_op=False):
    """
    Gather tensors from all ranks into a list with MPS compatibility.
    
    Collects tensors from all distributed training ranks into a list on each rank.
    Each rank contributes one tensor and receives a list containing tensors from
    all ranks. Useful for collecting distributed results or metrics.
    
    ## MPS Compatibility Handling
    
    Special handling required due to dual tensor parameters:
    1. Check both tensor_list and input tensor for MPS devices
    2. Move any MPS tensors to CPU before gather operation
    3. Perform standard PyTorch distributed all_gather on CPU
    4. Copy results back to MPS devices maintaining tensor identities
    
    For non-MPS tensors (CUDA, CPU), operates as standard PyTorch all_gather.
    
    ## Usage Patterns
    
    ### Metric Collection
    ```python
    local_loss = torch.tensor([loss_value])
    all_losses = [torch.zeros_like(local_loss) for _ in range(world_size)]
    all_gather(all_losses, local_loss)
    # all_losses now contains loss from each rank
    ```
    
    ### Parameter Collection
    ```python
    local_param = model.some_parameter
    all_params = [torch.zeros_like(local_param) for _ in range(world_size)]
    all_gather(all_params, local_param)
    # all_params contains parameter from each rank
    ```
    
    ## Performance Considerations
    
    - **Memory Usage**: Creates copies of tensor for each rank
    - **Communication Volume**: Scales with number of ranks and tensor size
    - **MPS Overhead**: Additional CPU transfers for Apple Silicon compatibility
    
    Args:
        tensor_list: List of tensors to populate with gathered results
        tensor: Local tensor to contribute to the gather operation
        group: Process group for communication (default: all ranks)
        async_op: Whether to perform asynchronous operation (default: False)
        
    Returns:
        None (tensor_list populated with results)
        
    Called by:
        Custom metric aggregation and debugging functions
        Research code for parameter analysis
        
    Calls:
        torch.distributed.all_gather() for actual communication
    """
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)


# @mps_compatible
# def reduce_scatter(tensor):
#     return dist.reduce_scatter(tensor)

# @mps_compatible
# def reduce(tensor):
#     return dist.reduce(tensor)

# @mps_compatible
# def gather(tensor):
#     return dist.gather(tensor)
