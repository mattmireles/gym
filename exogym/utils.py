"""
ExoGym Utils - Configuration Extraction and Logging Utilities

This module provides utilities for safely extracting configuration from complex
objects for logging purposes, particularly designed for AI-compatible serialization
and comprehensive model/training metadata collection.

## Core Functionality

### LogModule
Base class that provides automatic configuration extraction for any object.
Classes inheriting from LogModule gain the ability to serialize their state
for logging to WandB, CSV files, or other monitoring systems.

### extract_config()
Recursive configuration extraction function designed to handle complex PyTorch
objects safely. The key challenge is that PyTorch training objects contain
many unpickleable elements (tensors, CUDA contexts, functions) that need to
be filtered out or converted to serializable representations.

## Serialization Safety

### Problematic PyTorch Objects
- **Tensors**: Contain GPU memory references, too large for logging
- **Modules**: Complex nested objects with unpickleable state  
- **Optimizers**: Contain internal CUDA state and function references
- **Functions/Methods**: Can't be serialized across process boundaries

### Safe Representations
- **Tensors**: Convert to shape strings like "<Tensor [1024, 768]>"
- **Modules**: Extract class name like "<Module TransformerBlock>"
- **Devices**: Convert to string representation like "cuda:0"
- **Complex Objects**: Recursively extract serializable attributes

## AI-First Design

This utility is specifically designed for AI developers and AI-assisted debugging:

### Comprehensive Metadata
- Extracts model parameter counts, layer types, and architectural details
- Captures training hyperparameters, device configurations, and strategy settings
- Provides complete training run reproducibility information

### Depth-Limited Recursion
- Prevents infinite loops in circular object references
- Limits object traversal depth to avoid overwhelming logs
- Focuses on most relevant configuration information

### Type-Aware Processing
- Handles PyTorch-specific types intelligently
- Preserves numeric precision for hyperparameters
- Converts complex objects to human-readable strings

## Key Functions

### create_config()
Builds comprehensive WandB/CSV-compatible configuration from training components:
- Model architecture and parameter information
- Strategy configuration and hyperparameters  
- Training node settings and device information
- Custom configuration overrides

### log_model_summary()
Extracts detailed model architecture information:
- Parameter counts (total, trainable, by layer type)
- Model configuration objects if available
- Layer type distribution and model structure

### safe_log_dict()
Converts arbitrary dictionaries to logging-safe format with configurable
key prefixes for namespace organization.

## Usage Patterns

### Automatic Configuration Extraction
```python
class MyStrategy(LogModule):
    def __init__(self, lr=0.001, batch_size=32):
        self.lr = lr
        self.batch_size = batch_size
    
    # Automatically gets __config__() method
```

### Model Analysis
```python
model_info = log_model_summary(model)
# Returns: {"total_params": 124M, "layer_types": {"Linear": 12, "LayerNorm": 6}}
```

### Training Configuration
```python
config = create_config(model, strategy, train_node)
# Returns complete training configuration for reproducibility
```

## Called by:
- Logger classes (WandbLogger, CSVLogger) for configuration serialization
- Strategy and TrainNode classes via LogModule inheritance
- Training scripts for run metadata collection

## Calls:
- Built-in Python introspection (hasattr, __dict__, type)
- PyTorch model analysis (parameters(), named_modules())
- Recursive object traversal with safety limits

## Performance Considerations

### Memory Efficiency
- Limits collection size (50 keys max, 10 list items max)
- Skips private attributes and large objects
- Uses lazy evaluation for expensive operations

### Safety Limits
- Maximum recursion depth prevents infinite loops
- Early termination for oversized collections
- Error handling prevents crashes during extraction

This utility enables comprehensive training run documentation while maintaining
safety and performance for production distributed training workloads.
"""

import torch
from typing import Any, Dict, List


class UtilsConstants:
    """
    Named constants for configuration extraction and logging utilities.
    
    These constants control the behavior of recursive configuration extraction
    and model analysis functions, providing performance limits and safety bounds.
    """
    
    # Configuration Extraction Limits
    MAX_RECURSION_DEPTH = 10
    """
    Maximum recursion depth for extract_config() function.
    
    Prevents infinite loops when extracting configuration from objects with
    circular references or deeply nested structures. The depth limit ensures
    that configuration extraction completes in reasonable time while capturing
    the most relevant information.
    
    Value chosen to handle typical nested object structures (Model → Strategy → 
    Optimizer → Scheduler) while preventing stack overflow from pathological cases.
    """
    
    MAX_LIST_ITEMS = 10
    """
    Maximum number of list/tuple items to extract during configuration processing.
    
    Large collections (datasets, parameter lists, etc.) are truncated to prevent
    overwhelming logs with excessive detail. The first 10 items usually provide
    sufficient information about collection structure and content patterns.
    
    Remaining items are indicated with "... (N more items)" notation.
    """
    
    MAX_DICT_KEYS = 50
    """
    Maximum number of dictionary keys to extract during configuration processing.
    
    Large dictionaries (model state dicts, large configurations) are truncated
    to prevent log overwhelming. 50 keys typically capture the most important
    configuration parameters while maintaining reasonable log size.
    
    Remaining keys are indicated with "... (truncated, N more keys)" notation.
    """


class LogModule:
    """
    Mixin class that provides automatic configuration extraction for any object.
    
    LogModule enables any Python class to automatically generate logging-safe
    configuration dictionaries by inheriting from this mixin. The configuration
    extraction is designed specifically for AI-first documentation and debugging,
    providing comprehensive metadata about object state while filtering out
    unpickleable or sensitive information.
    
    ## Core Functionality
    
    ### Automatic Config Generation
    - Recursively extracts all serializable attributes from object
    - Filters out private attributes, functions, and unpickleable objects
    - Converts PyTorch-specific objects to safe string representations
    - Provides consistent configuration format across all ExoGym components
    
    ### AI-First Design
    - Generates metadata specifically designed for AI developer consumption
    - Includes type information, shape details, and architectural summaries
    - Enables comprehensive experiment tracking and reproducibility
    - Supports debugging and configuration comparison workflows
    
    ## Usage Pattern
    
    Classes inherit from LogModule to gain automatic configuration extraction:
    ```python
    class MyStrategy(LogModule):
        def __init__(self, lr=0.001, batch_size=32):
            self.lr = lr
            self.batch_size = batch_size
            # Other initialization...
    
    strategy = MyStrategy(lr=0.01)
    config = strategy.__config__()  # Automatically extracts configuration
    ```
    
    ## Integration with ExoGym
    
    ### Strategy Classes
    - All Strategy subclasses inherit from LogModule
    - Enables automatic strategy configuration extraction for logging
    - Supports configuration comparison and hyperparameter tracking
    
    ### TrainNode Integration
    - TrainNode inherits from LogModule for training state extraction
    - Provides comprehensive training configuration documentation
    - Enables debugging and experiment reproducibility
    
    ### Logger Integration
    - Logger classes use LogModule configurations for metadata extraction
    - Supports both WandB and CSV logging with consistent configuration format
    - Enables automatic experiment documentation and tracking
    
    Called by:
        Logger classes for configuration extraction during initialization
        Debugging and analysis tools for object introspection
        
    Uses:
        extract_config() function for recursive configuration extraction
    """
    
    def __config__(self, remove_keys: List[str] = None):
        """
        Extract complete configuration dictionary from this object.
        
        Recursively extracts all serializable attributes and converts them
        to a logging-safe format. Optionally removes specified keys for
        custom configuration filtering.
        
        ## Extraction Process
        
        1. **Recursive Traversal**: Walks through all object attributes
        2. **Type Conversion**: Converts PyTorch objects to safe representations
        3. **Filtering**: Removes private attributes and unpickleable objects
        4. **Sanitization**: Converts complex objects to string descriptions
        
        ## Configuration Content
        
        The extracted configuration includes:
        - Scalar hyperparameters (learning rates, batch sizes, etc.)
        - Object type information and class names
        - Tensor shapes and device information
        - Nested object configurations (recursive extraction)
        
        Args:
            remove_keys: Optional list of keys to exclude from configuration
            
        Returns:
            dict: Complete configuration dictionary with serializable values
            
        Called by:
            create_config() function for logger initialization
            Debugging tools for object state inspection
            
        Example:
            ```python
            strategy = DiLoCoStrategy(H=100, lr=0.001)
            config = strategy.__config__()
            # Returns: {"H": 100, "lr": 0.001, "class": "DiLoCoStrategy", ...}
            ```
        """
        # Extract complete configuration using recursive extraction
        config = extract_config(self)

        # Remove specified keys if provided
        if remove_keys:
            for key in remove_keys:
                if key in config:
                    del config[key]

        return config


def extract_config(obj, max_depth=UtilsConstants.MAX_RECURSION_DEPTH, current_depth=0):
    """
    Recursively extract configuration from any Python object with PyTorch safety.
    
    This function performs deep introspection of Python objects to extract
    serializable configuration information. It's specifically designed to handle
    PyTorch objects safely and provide meaningful representations for AI development
    and debugging workflows.
    
    ## Core Algorithm
    
    ### Recursive Traversal
    - Walks through object attributes using __dict__ introspection
    - Follows object references up to configurable depth limit
    - Handles circular references and infinite recursion gracefully
    
    ### Type-Aware Processing
    - **Basic Types**: Pass through scalars unchanged (int, float, str, bool)
    - **PyTorch Objects**: Convert to safe string representations
    - **Collections**: Process lists/dicts with size limits for performance
    - **Custom Objects**: Recursively extract __dict__ attributes
    
    ## PyTorch Object Handling
    
    ### Tensor Representation
    - Avoids memory issues from large tensor serialization
    - Provides shape and type information for debugging
    - Enables model architecture documentation
    
    ### Module Representation  
    - Extracts module class names and basic info
    - Avoids serializing complex module state
    - Preserves architectural information
    
    ## Performance and Safety Limits
    
    ### Depth Limiting
    - Maximum recursion depth prevents infinite loops
    - Configurable depth limit (default: 10 levels)
    - Circular reference protection
    
    ### Collection Size Limits
    - Lists: Truncated to first 10 items
    - Dictionaries: Limited to 50 keys
    - Prevents overwhelming logs with large data structures
    
    Args:
        obj: Python object to extract configuration from
        max_depth: Maximum recursion depth (default: 10)
        current_depth: Current recursion level (internal use)
        
    Returns:
        dict|str|scalar: Extracted configuration in serializable format
        
    Called by:
        LogModule.__config__() for automatic configuration extraction
        create_config() for logger initialization
        Debugging and analysis tools
    """
    if current_depth >= max_depth:
        return str(type(obj).__name__)

    if obj is None:
        return None

    # Handle primitive types
    if isinstance(obj, (int, float, str, bool)):
        return obj

    # Handle sequences (but avoid strings which are also sequences)
    if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
        return [
            extract_config(item, max_depth, current_depth + 1) for item in obj[:UtilsConstants.MAX_LIST_ITEMS]
        ]  # Limit to first MAX_LIST_ITEMS items

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(key, str) and len(result) < UtilsConstants.MAX_DICT_KEYS:  # Limit number of keys
                result[key] = extract_config(value, max_depth, current_depth + 1)
        return result

    if isinstance(obj, torch.device):
        return obj.__str__()

    # Skip unpickleable types
    if isinstance(
        obj,
        (
            torch.Tensor,
            torch.nn.Module,
            torch.optim.Optimizer,
            torch.nn.Parameter,
            torch.dtype,
        ),
    ):
        if isinstance(obj, torch.Tensor):
            return f"<Tensor {list(obj.shape)}>"
        elif isinstance(obj, torch.nn.Module):
            return f"<Module {type(obj).__name__}>"
        elif isinstance(obj, torch.optim.Optimizer):
            return f"<Optimizer {type(obj).__name__}>"
        else:
            return f"<{type(obj).__name__}>"

    # Skip functions, methods, and other callables
    if callable(obj):
        return f"<function {getattr(obj, '__name__', 'unknown')}>"

    # Handle objects with __dict__ (like config objects)
    if hasattr(obj, "__dict__"):
        result = {}
        for key, value in obj.__dict__.items():
            if (
                not key.startswith("_") and len(result) < UtilsConstants.MAX_DICT_KEYS
            ):  # Skip private attributes
                result[key] = extract_config(
                    value, max_depth, current_depth + 1
                )
        return result

    # For other objects, try to get basic info
    if type(obj) in [float, int, str, bool]:
        return obj
    else:
        return f"<{type(obj).__name__}>"


def create_config(
    model: torch.nn.Module, strategy, train_node, extra_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create comprehensive configuration dictionary for experiment tracking.
    
    This function aggregates configuration information from all major training
    components into a single, comprehensive dictionary suitable for logging
    to WandB, CSV files, or other experiment tracking systems. It provides
    complete experiment reproducibility information.
    
    ## Configuration Sources
    
    ### Strategy Configuration
    - Extracts hyperparameters, optimizer settings, and algorithm parameters
    - Includes communication patterns, learning rate schedules, and timing
    - Provides complete strategy state for experiment reproduction
    
    ### TrainNode Configuration
    - Training loop configuration including batch sizes and intervals
    - Dataset information and data loading parameters
    - Hardware and device configuration details
    - Evaluation and checkpointing settings
    
    ### Model Architecture
    - Model class name and architecture identifier
    - Parameter count (in millions) for size comparison
    - Model-specific configuration if available via get_num_params()
    
    ### Extra Configuration
    - Custom experiment parameters and metadata
    - Environment variables and system information
    - Run-specific identifiers and tracking information
    
    ## Model Parameter Counting
    
    ### Automatic Parameter Detection
    - Uses model.get_num_params() if available (custom method)
    - Fallback to sum(p.numel() for p in model.parameters())
    - Converts to millions for human-readable format
    
    ### Parameter Count Benefits
    - Enables model size comparison across experiments
    - Supports resource planning and scaling decisions
    - Provides architectural complexity metrics
    
    ## Configuration Merging Strategy
    
    ### Hierarchical Structure
    1. **Strategy Config**: Core training algorithm configuration
    2. **TrainNode Config**: Training loop and infrastructure settings
    3. **Model Config**: Architecture and parameter information
    4. **Extra Config**: Custom experiment metadata
    
    ### Key Conflict Resolution
    - TrainNode config updates (overwrites) strategy config for shared keys
    - Extra config takes highest precedence
    - Enables flexible configuration override patterns
    
    ## Usage in ExoGym
    
    ### Logger Integration
    - WandB logger uses this for comprehensive experiment tracking
    - CSV logger persists complete config to JSON files
    - Enables experiment comparison and analysis
    
    ### Reproducibility
    - Contains all information needed to reproduce experiments
    - Supports automated experiment replication
    - Enables systematic hyperparameter study tracking
    
    Args:
        model: PyTorch model being trained
        strategy: Training strategy with hyperparameters
        train_node: Training node with infrastructure configuration
        extra_config: Additional custom configuration dictionary
        
    Returns:
        dict: Complete configuration dictionary with all training metadata
        
    Called by:
        WandbLogger.__init__() for experiment tracking setup
        CSVLogger.__init__() for configuration persistence
        Debugging and analysis tools for configuration extraction
        
    Example:
        ```python
        config = create_config(
            model=gpt_model,
            strategy=diloco_strategy,
            train_node=train_node,
            extra_config={"experiment_name": "scaling_study", "notes": "testing H=200"}
        )
        # Returns comprehensive config with model, strategy, training, and custom info
        ```
    """
    wandb_config = {}

    # Extract strategy configuration (hyperparameters, algorithm settings)
    wandb_config["strategy"] = strategy.__config__()
    
    # Extract and merge training node configuration (infrastructure, training loop)
    wandb_config.update(train_node.__config__())

    # Extract model architecture information
    if model:
        wandb_config.update(
            {
                "model_name": model.__class__.__name__,
                # Note: model_config extraction disabled to avoid large configs
                # "model_config": extract_config(model),
            }
        )

        # Calculate model parameter count for size comparison
        if hasattr(model, "get_num_params"):
            # Use custom parameter counting method if available
            wandb_config["model_parameters"] = model.get_num_params() / 1e6
        else:
            # Fallback to standard PyTorch parameter counting
            wandb_config["model_parameters"] = (
                sum(p.numel() for p in model.parameters()) / 1e6
            )

    # Include additional custom configuration
    if extra_config:
        for key, value in extra_config.items():
            wandb_config[key] = extract_config(value)

    return wandb_config


def log_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Extract comprehensive model architecture summary for logging and analysis.
    
    This function provides detailed model architecture information suitable for
    experiment tracking, debugging, and performance analysis. It extracts both
    quantitative metrics (parameter counts, layer statistics) and qualitative 
    information (architecture details) in a format optimized for AI developer consumption.
    
    ## Summary Categories
    
    ### Basic Model Information
    - Model class name and module path for architecture identification
    - Module hierarchy and component structure information
    - Architecture family classification and source identification
    
    ### Parameter Analysis
    - Total parameter count using model.get_num_params() or fallback counting
    - Trainable parameter count (requires_grad=True parameters only)
    - Parameter counts in both absolute numbers and millions for readability
    - Memory usage implications and model size estimates
    
    ### Layer Type Distribution
    - Counts of each layer type (Linear, Conv2d, LayerNorm, etc.)
    - Architecture composition analysis
    - Enables architectural pattern recognition and comparison
    
    ### Configuration Extraction
    - Model-specific configuration objects if available (model.config)
    - Architectural hyperparameters and design choices
    - Safe serialization of complex configuration objects
    
    Args:
        model: PyTorch model to analyze and summarize
        
    Returns:
        dict: Comprehensive model summary with architecture and parameter information
        
    Called by:
        create_config() for comprehensive experiment configuration
        Logger initialization for model architecture documentation
        Debugging and analysis tools for model introspection
    """
    summary = {
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }

    try:
        # Parameter count
        if hasattr(model, "get_num_params"):
            summary["total_params"] = model.get_num_params()
        else:
            summary["total_params"] = sum(p.numel() for p in model.parameters())

        summary["total_params_M"] = summary["total_params"] / 1e6

        # Trainable parameters
        summary["trainable_params"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        summary["trainable_params_M"] = summary["trainable_params"] / 1e6

        # Model config if available
        if hasattr(model, "config"):
            summary["config"] = extract_config(model.config)

        # Layer information
        layer_types = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type != model.__class__.__name__:  # Skip the root module
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
        summary["layer_types"] = layer_types

    except Exception as e:
        summary["error"] = f"Error extracting model summary: {str(e)}"

    return summary


def safe_log_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Convert a dictionary to a wandb-safe format.

    Args:
      data: Dictionary to convert
      prefix: Prefix to add to keys

    Returns:
      dict: Wandb-safe dictionary
    """
    safe_dict = {}

    for key, value in data.items():
        safe_key = f"{prefix}_{key}" if prefix else key
        safe_dict[safe_key] = extract_config(value)

    return safe_dict
