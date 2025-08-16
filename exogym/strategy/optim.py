"""
ExoGym OptimSpec - Optimizer Specification and Factory Pattern

This module implements a standardized optimizer specification pattern that enables
flexible, configurable optimizer creation across the distributed training framework.
The OptimSpec pattern provides type-safe optimizer construction with runtime
flexibility for experimentation and configuration management.

## Core Design Pattern: Factory with Configuration

### OptimSpec Factory Pattern
The OptimSpec class implements the factory pattern for PyTorch optimizers:
- **Class Storage**: Stores optimizer class reference (not instance)
- **Parameter Storage**: Stores optimizer configuration as dictionary
- **Lazy Construction**: Creates optimizer instance only when model is available
- **Device Awareness**: Automatically handles model parameter device placement

### String-Based Optimizer Selection
Enables configuration-driven optimizer selection for experiments:
- JSON/YAML configuration files can specify optimizers by name
- Command-line interfaces can accept optimizer strings
- Hyperparameter search can vary optimizers systematically
- A/B testing of optimizer choices becomes straightforward

## Key Benefits for Distributed Training

### Strategy Integration
All distributed training strategies use OptimSpec for optimizer management:
- **DiLoCoStrategy**: Uses separate inner/outer optimizer specs
- **SimpleReduceStrategy**: Uses single optimizer spec for gradient averaging
- **Custom Strategies**: Can define complex multi-optimizer configurations

### Multiprocessing Compatibility
OptimSpec enables safe optimizer transfer across process boundaries:
- **Serializable Configuration**: Can be pickled and sent to worker processes
- **Late Binding**: Optimizer instance created after model is available on target device
- **No CUDA Context Issues**: Avoids GPU context transfer problems in multiprocessing

### Hardware Agnostic Optimization
OptimSpec handles device placement automatically:
- **Model Parameter Binding**: Automatically binds to model.parameters() on correct device
- **CUDA/MPS/CPU Compatibility**: Works transparently across device types
- **Memory Efficiency**: No duplicate optimizer states across devices

## Architecture Integration

### Strategy Construction Pattern
```python
# Strategy defines optimizer requirement
class MyStrategy(Strategy):
    def __init__(self, optim_spec=None):
        self.optim_spec = ensure_optim_spec(optim_spec)  # Flexible input handling
    
    def _init_node(self, model, rank, num_nodes):
        self.optim = self.optim_spec.build(model)  # Create on target device
```

### Optimizer Selection Flexibility
```python
# String-based selection for configuration files
strategy = MyStrategy(optim_spec="adamw")

# Direct class specification for programmatic use
strategy = MyStrategy(optim_spec=OptimSpec(torch.optim.SGD, lr=0.01))

# Automatic default for simple cases
strategy = MyStrategy()  # Uses AdamW with sensible defaults
```

## Common Optimizer Configurations

### Modern Training Defaults
- **AdamW**: Default choice for transformer models and modern architectures
- **Decoupled Weight Decay**: Better regularization than L2 penalty
- **Adaptive Learning Rate**: Handles varying gradient scales automatically

### Classical Training Options
- **SGD with Momentum**: Traditional choice for CNN training
- **Learning Rate Scheduling**: Often requires manual LR schedule tuning
- **Nesterov Momentum**: Improved convergence for some optimization landscapes

### Specialized Optimizers
- **RMSprop**: Good for RNN training and non-stationary objectives
- **Adagrad**: Suitable for sparse gradient scenarios
- **Custom Optimizers**: Easy to integrate via class reference

## Memory and Performance Considerations

### Optimizer State Management
Different optimizers have varying memory footprints:
- **SGD**: Minimal state (momentum buffer only)
- **AdamW**: Moderate state (first and second moment estimates)
- **Adagrad**: Growing state (accumulates squared gradients)

### Device Memory Optimization
OptimSpec enables optimizer memory optimization:
- **Delayed Construction**: Optimizer created only when needed
- **Device-Local State**: Optimizer state stays on training device
- **No Cross-Device Transfers**: Avoids expensive memory copies

## Called by:
- Strategy classes during _init_node() for optimizer creation
- Training configuration parsers for string-to-optimizer conversion
- Hyperparameter search systems for systematic optimizer variation

## Calls:
- torch.optim.* classes for actual optimizer construction
- model.parameters() for parameter binding during build()

## Usage Patterns:

### Basic Optimizer Specification
```python
# Simple specification with default parameters
optim_spec = OptimSpec(torch.optim.AdamW)

# Specification with custom parameters
optim_spec = OptimSpec(torch.optim.SGD, lr=0.01, momentum=0.9)
```

### Configuration-Driven Selection
```python
# From configuration files or command line
optim_spec = OptimSpec.from_string("adamw", lr=3e-4, weight_decay=0.01)

# Flexible input handling
optim_spec = ensure_optim_spec("sgd", lr=0.01)  # String input
optim_spec = ensure_optim_spec(OptimSpec(torch.optim.Adam))  # Direct input
optim_spec = ensure_optim_spec(None)  # Uses AdamW default
```

### Multi-Optimizer Strategies
```python
# DiLoCo example with different inner/outer optimizers
class DiLoCoStrategy(Strategy):
    def __init__(self, inner_optim="adamw", outer_optim="sgd"):
        self.inner_spec = ensure_optim_spec(inner_optim)
        self.outer_spec = ensure_optim_spec(outer_optim, lr=0.7, momentum=0.9)
```

This module provides the foundation for flexible, efficient optimizer management
throughout the ExoGym distributed training framework, enabling both ease of use
and advanced optimization experimentation.
"""

import torch

from typing import Type, Union, Optional

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OptimSpec:
    """
    Optimizer specification factory for configurable optimizer construction.
    
    OptimSpec implements the factory pattern for PyTorch optimizers, enabling
    flexible, configurable optimizer creation that's compatible with distributed
    training, multiprocessing, and serialization requirements.
    
    ## Factory Pattern Implementation
    
    ### Configuration Storage
    - **Optimizer Class**: Stores reference to PyTorch optimizer class (not instance)
    - **Parameter Dictionary**: Stores constructor arguments as serializable dictionary
    - **Lazy Construction**: Creates optimizer instances only when model is available
    
    ### Benefits of Factory Pattern
    - **Serialization Safe**: Can be pickled and transferred across process boundaries
    - **Device Agnostic**: Optimizer created on target device with correct model parameters
    - **Memory Efficient**: No premature optimizer state allocation
    
    ## Usage in Distributed Training
    
    ### Strategy Integration Pattern
    ```python
    class DiLoCoStrategy(Strategy):
        def __init__(self, optim_spec=None):
            # Store specification, not optimizer instance
            self.optim_spec = ensure_optim_spec(optim_spec)
        
        def _init_node(self, model, rank, num_nodes):
            # Create optimizer on worker process with model on correct device
            self.optim = self.optim_spec.build(model)
    ```
    
    ### Multiprocessing Compatibility
    - Configuration survives torch.multiprocessing.spawn()
    - No CUDA context transfer issues
    - Optimizer state created fresh on each worker device
    
    ## Configuration Flexibility
    
    ### Constructor Patterns
    ```python
    # Basic usage with defaults
    spec = OptimSpec(torch.optim.AdamW)
    
    # With hyperparameters
    spec = OptimSpec(torch.optim.SGD, lr=0.01, momentum=0.9)
    
    # String-based construction
    spec = OptimSpec.from_string("adamw", lr=3e-4, weight_decay=0.01)
    ```
    
    ### Parameter Management
    - **Keyword Arguments**: All optimizer parameters stored as kwargs
    - **Runtime Binding**: Parameters applied during build() call
    - **Override Support**: Can merge additional parameters at build time
    
    Attributes:
        cls: PyTorch optimizer class reference
        kwargs: Dictionary of constructor arguments for the optimizer
    """
    cls: Type[torch.optim.Optimizer] = torch.optim.AdamW
    kwargs: Dict[str, Any] = None  # e.g. {'lr': 3e-4}

    def __init__(self, cls: Type[torch.optim.Optimizer], **kwargs: Dict[str, Any]):
        """
        Initialize optimizer specification with class and parameters.
        
        Args:
            cls: PyTorch optimizer class (e.g., torch.optim.AdamW)
            **kwargs: Optimizer constructor arguments (lr, weight_decay, etc.)
        """
        self.cls = cls
        self.kwargs = kwargs

    @classmethod
    def from_string(cls, name: str, **kwargs) -> "OptimSpec":
        """
        Create OptimSpec from optimizer name string for configuration-driven selection.
        
        This factory method enables string-based optimizer specification, which is
        essential for configuration files, command-line interfaces, and hyperparameter
        search systems. It provides a standardized mapping from optimizer names to
        PyTorch optimizer classes.
        
        ## Supported Optimizers
        
        ### Modern Adaptive Optimizers
        - **"adamw"**: AdamW with decoupled weight decay (recommended default)
        - **"adam"**: Original Adam optimizer
        - **"rmsprop"**: RMSprop for non-stationary objectives
        
        ### Classical Optimizers
        - **"sgd"**: Stochastic Gradient Descent (often with momentum)
        - **"adagrad"**: Adagrad for sparse gradients and online learning
        
        ## Configuration Integration
        
        ### JSON/YAML Configuration Files
        ```json
        {
            "strategy": {
                "optimizer": "adamw",
                "learning_rate": 3e-4,
                "weight_decay": 0.01
            }
        }
        ```
        
        ### Command Line Interfaces
        ```bash
        python train.py --optimizer sgd --lr 0.01 --momentum 0.9
        ```
        
        ### Hyperparameter Search
        ```python
        optimizers = ["adam", "adamw", "sgd"]
        for opt_name in optimizers:
            spec = OptimSpec.from_string(opt_name, lr=search_lr)
        ```
        
        ## Error Handling
        
        ### Invalid Optimizer Names
        - Provides clear error message with available options
        - Case-insensitive matching for user convenience
        - Suggests valid alternatives when unknown optimizer specified
        
        Args:
            name: Optimizer name string (case-insensitive)
            **kwargs: Optimizer constructor arguments (lr, momentum, etc.)
            
        Returns:
            OptimSpec: Configured optimizer specification
            
        Raises:
            ValueError: If optimizer name is not recognized
            
        Called by:
            ensure_optim_spec() for flexible input handling
            Configuration parsers for string-to-optimizer conversion
            Hyperparameter search systems for systematic variation
            
        Example:
            ```python
            # Basic string-based construction
            spec = OptimSpec.from_string("adamw")
            
            # With hyperparameters
            spec = OptimSpec.from_string("sgd", lr=0.01, momentum=0.9, nesterov=True)
            
            # Case-insensitive matching
            spec = OptimSpec.from_string("AdamW", lr=3e-4)  # Works fine
            ```
        """
        # Standard optimizer name mapping for configuration systems
        optimizer_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        name_lower = name.lower()
        if name_lower not in optimizer_map:
            available = ", ".join(optimizer_map.keys())
            raise ValueError(
                f"Unknown optimizer '{name}'. Available options: {available}"
            )

        return cls(optimizer_map[name_lower], **kwargs)

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Construct optimizer instance bound to model parameters.
        
        This method implements the core factory functionality by creating an actual
        PyTorch optimizer instance with the stored configuration. It's called during
        strategy initialization on each worker process after the model is available
        on the target device.
        
        ## Lazy Construction Benefits
        
        ### Device Placement
        - Optimizer created after model is moved to target device (CUDA/MPS/CPU)
        - Optimizer state automatically allocated on correct device
        - No manual device transfer required for optimizer state
        
        ### Memory Efficiency
        - Optimizer state allocated only when needed
        - No premature memory allocation during strategy configuration
        - Enables memory-efficient strategy comparison and experimentation
        
        ### Multiprocessing Safety
        - Optimizer instance created fresh on each worker process
        - No CUDA context transfer issues across process boundaries
        - Avoids shared state problems in distributed training
        
        ## Implementation Details
        
        ### Parameter Binding
        - Automatically binds to model.parameters() generator
        - Includes all trainable parameters (requires_grad=True)
        - Respects parameter device placement and dtype
        
        ### Configuration Application
        - Applies stored kwargs as optimizer constructor arguments
        - Handles None kwargs gracefully (empty dict fallback)
        - Preserves all hyperparameter settings from specification
        
        ## Usage in Distributed Training
        
        ### Strategy Initialization Pattern
        ```python
        def _init_node(self, model, rank, num_nodes):
            # Model already on correct device at this point
            self.optim = self.optim_spec.build(model)  # Creates optimizer on same device
        ```
        
        ### Device Compatibility
        - Works transparently with CUDA, MPS, and CPU backends
        - No special handling required for different device types
        - Optimizer state follows model device placement automatically
        
        Args:
            model: PyTorch model with parameters on target device
            
        Returns:
            torch.optim.Optimizer: Configured optimizer instance bound to model
            
        Called by:
            Strategy._init_node() methods during worker process initialization
            Testing code for optimizer validation and experimentation
            
        Example:
            ```python
            # Create specification
            spec = OptimSpec(torch.optim.AdamW, lr=3e-4, weight_decay=0.01)
            
            # Later, when model is available on target device
            model = model.to(device)  # Move to CUDA/MPS/CPU
            optimizer = spec.build(model)  # Creates optimizer on same device
            ```
        """
        return self.cls(model.parameters(), **(self.kwargs or {}))


def ensure_optim_spec(
    optim: Union[str, OptimSpec, None], default: Optional[OptimSpec] = None, **kwargs
) -> OptimSpec:
    """
    Convert flexible optimizer input to standardized OptimSpec instance.
    
    This utility function provides a unified interface for optimizer specification,
    accepting multiple input types and converting them to OptimSpec instances.
    It's designed to make strategy constructors more user-friendly while maintaining
    type safety and configuration consistency.
    
    ## Input Type Handling
    
    ### None Input (Default Behavior)
    - Returns provided default OptimSpec if available
    - Falls back to AdamW with sensible defaults
    - Applies any additional kwargs to the default configuration
    
    ### String Input (Configuration-Driven)
    - Converts optimizer name to OptimSpec via from_string()
    - Enables JSON/YAML configuration file integration
    - Supports case-insensitive optimizer names
    
    ### OptimSpec Input (Pass-Through)
    - Returns existing OptimSpec unchanged if no additional kwargs
    - Merges additional kwargs if provided (parameter override pattern)
    - Preserves original configuration while allowing customization
    
    ## Use Cases in ExoGym
    
    ### Strategy Constructor Flexibility
    ```python
    class MyStrategy(Strategy):
        def __init__(self, optim_spec=None):
            # Accepts any input type, converts to OptimSpec
            self.optim_spec = ensure_optim_spec(optim_spec)
    
    # All these work:
    strategy1 = MyStrategy()  # Uses AdamW default
    strategy2 = MyStrategy("sgd")  # String-based
    strategy3 = MyStrategy(OptimSpec(torch.optim.Adam))  # Direct spec
    ```
    
    ### Configuration Override Pattern
    ```python
    base_spec = OptimSpec(torch.optim.SGD, lr=0.01)
    # Override learning rate while preserving optimizer type
    new_spec = ensure_optim_spec(base_spec, lr=0.001)
    ```
    
    ### Hyperparameter Search Integration
    ```python
    def create_strategy(optimizer_choice):
        # optimizer_choice could be string, spec, or None
        spec = ensure_optim_spec(optimizer_choice, lr=search_lr)
        return MyStrategy(optim_spec=spec)
    ```
    
    ## Parameter Merging Logic
    
    ### Precedence Rules
    1. **Additional kwargs**: Highest precedence (override everything)
    2. **Existing OptimSpec kwargs**: Medium precedence
    3. **Default configuration**: Lowest precedence
    
    ### Safe Merging
    - Dictionary update preserves existing keys unless overridden
    - None kwargs handling prevents errors
    - Type validation ensures correct input types
    
    Args:
        optim: Optimizer specification (string, OptimSpec, or None)
        default: Default OptimSpec to use when optim is None
        **kwargs: Additional optimizer parameters to apply/override
        
    Returns:
        OptimSpec: Standardized optimizer specification
        
    Raises:
        TypeError: If optim is not a supported type
        ValueError: If string optimizer name is not recognized
        
    Called by:
        Strategy constructors for flexible optimizer specification
        Configuration parsers for optimizer input validation
        Testing code for systematic optimizer variation
        
    Examples:
        ```python
        # Default behavior
        spec = ensure_optim_spec(None)  # AdamW with defaults
        
        # String conversion
        spec = ensure_optim_spec("sgd", lr=0.01)  # SGD with custom LR
        
        # OptimSpec pass-through
        base = OptimSpec(torch.optim.Adam, lr=0.001)
        spec = ensure_optim_spec(base)  # Unchanged
        
        # Parameter override
        spec = ensure_optim_spec(base, lr=0.01)  # Adam with new LR
        
        # Custom default
        default = OptimSpec(torch.optim.RMSprop, lr=0.01)
        spec = ensure_optim_spec(None, default=default)  # Uses RMSprop
        ```
    """
    if optim is None:
        if default is None:
            # Standard default: AdamW with additional kwargs applied
            return OptimSpec(torch.optim.AdamW, **kwargs)
        else:
            # Use provided default, potentially modified by kwargs
            if kwargs:
                merged_kwargs = {**(default.kwargs or {}), **kwargs}
                return OptimSpec(default.cls, **merged_kwargs)
            return default
    elif isinstance(optim, str):
        # Convert string to OptimSpec with additional kwargs
        return OptimSpec.from_string(optim, **kwargs)
    elif isinstance(optim, OptimSpec):
        # Pass through OptimSpec, optionally merging additional kwargs
        if kwargs:
            merged_kwargs = {**(optim.kwargs or {}), **kwargs}
            return OptimSpec(optim.cls, **merged_kwargs)
        return optim
    else:
        raise TypeError(f"Expected str, OptimSpec, or None, got {type(optim)}")
