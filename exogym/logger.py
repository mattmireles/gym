"""
ExoGym Logger - Training Metrics and Progress Tracking

This module provides comprehensive logging infrastructure for distributed training
with support for both cloud-based (WandB) and local (CSV) logging backends.
The logging system is designed to handle the complexities of distributed training
including rank-specific responsibilities and metric aggregation.

## Logging Architecture

### Multi-Backend Support
- **WandbLogger**: Cloud-based experiment tracking with rich visualizations
- **CSVLogger**: Local file-based logging for offline analysis and CI/CD
- **Logger Base Class**: Common interface and progress bar management

### Distributed Training Integration
- **Rank-Aware Logging**: Only rank 0 performs actual logging to avoid conflicts
- **Metric Aggregation**: Handles local vs. global model evaluation correctly
- **Progress Synchronization**: Maintains consistent progress bars across processes

## Key Components

### Logger (Base Class)
Provides core functionality shared across all logging backends:
- **Progress Tracking**: tqdm-based progress bars with loss/LR display
- **Step Management**: Maintains training step counter for metric alignment
- **Interface Definition**: Abstract methods for metric logging

### WandbLogger  
Production-grade experiment tracking with cloud integration:
- **Configuration Logging**: Automatic extraction and upload of training configs
- **Rich Metrics**: Loss, perplexity, learning rate, and custom metrics
- **Run Management**: Automatic run naming, resumption, and artifact tracking
- **Team Collaboration**: Shared experiment visibility and comparison

### CSVLogger
Lightweight local logging for development and CI environments:
- **File Organization**: Structured directory layout with separate train/validation files
- **Configuration Persistence**: JSON export of complete training configuration
- **Offline Analysis**: CSV format compatible with pandas, Excel, and analysis tools
- **Update Handling**: Graceful handling of metric updates at same training step

## Metric Types and Responsibilities

### Training Metrics (All Ranks → Rank 0)
- **Train Loss**: Raw loss values from local model forward passes
- **Learning Rate**: Current optimizer learning rate from scheduler
- **Step Counter**: Global training step across all distributed processes

### Validation Metrics (Rank 0 & 1 → Rank 0)
- **Local Evaluation**: Rank 0 evaluates its local model state
- **Global Evaluation**: Rank 1 evaluates averaged model across all nodes
- **Perplexity Calculation**: Automatic exp(loss) conversion for language models

## Configuration Integration

### Automatic Config Extraction
Uses utils.create_config() to extract comprehensive metadata:
- **Model Architecture**: Parameter counts, layer types, model configuration
- **Training Setup**: Batch sizes, learning rates, hardware configuration
- **Strategy Details**: Communication patterns, optimization hyperparameters
- **Reproducibility**: Seeds, versions, device information

### WandB Integration
- **Project Organization**: Configurable project names for experiment grouping
- **Run Naming**: Custom run names with automatic fallback to timestamps
- **Config Upload**: Complete training configuration automatically uploaded
- **Resume Support**: Automatic run resumption for interrupted training

## Learning Rate Callback System

### Strategy Integration
- **Callback Registration**: Strategies register LR change callbacks with loggers
- **Automatic Updates**: LR changes automatically logged without manual intervention
- **Scheduler Compatibility**: Works with all PyTorch LR schedulers

### Display Integration
- **Progress Bar**: Real-time LR display in training progress bar
- **Metric Logging**: LR values logged alongside loss metrics for analysis

## File Organization (CSVLogger)

```
logs/
└── run_name/
    ├── config.json          # Complete training configuration
    ├── train.csv           # Training metrics (loss, LR, step)
    └── validation.csv      # Validation metrics (local/global loss)
```

## Error Handling and Robustness

### Import Safety
- **Optional Dependencies**: WandB import wrapped with helpful error messages
- **Graceful Degradation**: Missing dependencies don't crash training

### File System Resilience
- **Directory Creation**: Automatic creation of log directories
- **Concurrent Access**: Safe handling of multiple processes writing logs
- **Update Conflicts**: Graceful handling of metric updates at same step

## Usage Patterns

### Automatic Logger Selection
```python
if wandb_project:
    logger = WandbLogger(model, max_steps, strategy, train_node, wandb_project, run_name)
else:
    logger = CSVLogger(model, max_steps, strategy, train_node, run_name)
```

### Metric Logging
```python
logger.log_train(loss.item())          # Training loss
logger.log_loss(val_loss, "local")     # Validation loss
logger.increment_step()                # Step counter
```

## Called by:
- TrainNode during training loop for metrics collection
- Strategy classes for learning rate callback registration
- Training scripts for logger initialization and configuration

## Calls:
- utils.create_config() for comprehensive configuration extraction
- WandB API for cloud experiment tracking
- CSV file operations for local logging
- tqdm for progress bar management

This logging infrastructure provides comprehensive training visibility while
maintaining performance and compatibility across different deployment environments.
"""

from tqdm import tqdm
import numpy as np
from torch import nn

from .utils import create_config

import json
import os
from datetime import datetime
import csv


class Logger:
    """
    Abstract base class for training progress tracking and metrics logging.
    
    Logger provides the common interface and functionality shared across all
    logging backends (WandB, CSV, etc.). It handles progress bar management,
    step counting, and defines the standard logging interface that all
    concrete loggers must implement.
    
    ## Core Responsibilities
    
    ### Progress Tracking
    - Maintains global step counter across distributed training
    - Manages tqdm progress bar with loss and learning rate display
    - Provides real-time visual feedback during training
    
    ### Logging Interface
    - Defines standard methods for training metrics (loss, perplexity, LR)
    - Separates local vs global model evaluation logging
    - Enables consistent logging across different backends
    
    ### Learning Rate Monitoring
    - Tracks current learning rate for progress bar display
    - Integrates with strategy LR callback system
    - Provides real-time LR change visualization
    
    ## Distributed Training Integration
    
    ### Rank 0 Responsibility
    - Only rank 0 should create loggers to avoid conflicts
    - Progress bar and step counting coordinated across processes
    - Centralizes all logging output and progress tracking
    
    ### Step Synchronization
    - Step counter managed independently of progress bar
    - Enables flexible step tracking for different strategies
    - Supports both local and global step coordination
    
    ## Implementation Details
    
    ### Progress Bar Management
    - Uses tqdm for cross-platform progress visualization
    - Automatically updates with training loss and learning rate
    - Provides clean terminal output with postfix information
    
    ### Learning Rate Integration
    - Stores current LR for progress bar display
    - Updated via log_lr() callback from strategy schedulers
    - Enables real-time monitoring of LR scheduling behavior
    
    ## Usage Pattern
    
    ```python
    logger = WandbLogger(model, max_steps, strategy, train_node, project="my-experiment")
    
    # Training loop
    for step in range(max_steps):
        loss = train_step()
        logger.log_train(loss.item())
        logger.increment_step()
    ```
    
    ## Subclass Requirements
    
    Concrete loggers must implement:
    - log(): General data logging for custom metrics
    - log_loss(): Validation loss with local/global distinction
    - Additional backend-specific setup and teardown
    
    Called by:
        TrainNode.train() for metrics tracking during training loops
        Strategy classes via LR callbacks for learning rate updates
        
    Subclassed by:
        WandbLogger, CSVLogger for specific logging backends
    """
    
    def __init__(self, model: nn.Module, max_steps: int):
        """
        Initialize base logger with model and training configuration.
        
        Sets up progress bar, step tracking, and learning rate monitoring.
        The progress bar provides immediate visual feedback while concrete
        loggers handle persistent metric storage.
        
        Args:
            model: PyTorch model being trained (used for configuration extraction)
            max_steps: Total training steps for progress bar configuration
        """
        self.model = model
        self.max_steps = max_steps

        # Initialize progress bar for real-time training feedback
        self.pbar = tqdm(total=self.max_steps, initial=0)

        tqdm.write("Logger initialized.")

        # Track training progress and learning rate state
        self.step = 0
        self.current_lr = 0

    def log(self, data: dict):
        """
        Log arbitrary data dictionary (abstract method).
        
        Concrete loggers must implement this method to handle custom
        metrics and data logging in their specific format.
        
        Args:
            data: Dictionary of metric names to values
        """
        pass

    def log_loss(self, loss: float, name: str):
        """
        Log validation loss with identifier (abstract method).
        
        Concrete loggers must implement this method to handle validation
        loss logging with appropriate naming (local/global distinction).
        
        Args:
            loss: Validation loss value
            name: Loss identifier ("local" or "global" typically)
        """
        pass

    def log_train(self, loss: float):
        """
        Log training loss and update progress bar.
        
        Updates the progress bar with current training loss and learning rate
        for real-time monitoring. Called after each training step to provide
        immediate feedback on training progress.
        
        Args:
            loss: Current training loss value (scalar)
        """
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )

    def increment_step(self):
        """
        Advance internal step counter.
        
        Maintains separate step counter from progress bar for flexible
        step tracking. Enables coordination with strategy step counting
        and supports different stepping patterns.
        """
        self.step += 1

    def log_lr(self, lr: float):
        """
        Update current learning rate for progress bar display.
        
        Called by strategy LR callbacks to update the stored learning rate.
        The updated LR is displayed in the progress bar postfix during
        subsequent log_train() calls.
        
        Args:
            lr: Current learning rate value from scheduler
        """
        self.current_lr = lr


class WandbLogger(Logger):
    """
    Weights & Biases (WandB) logger for cloud-based experiment tracking.
    
    WandbLogger provides production-grade experiment tracking with rich
    visualizations, team collaboration, and comprehensive configuration logging.
    It automatically extracts training configuration and provides real-time
    metrics streaming to the WandB cloud platform.
    
    ## Key Features
    
    ### Cloud Integration
    - Automatic upload of training metrics to WandB cloud
    - Rich web interface with loss curves, system metrics, and model comparisons
    - Team collaboration with shared experiment visibility
    - Automatic experiment versioning and reproducibility tracking
    
    ### Resume Capability
    - Automatic run resumption with "resume": "allow" configuration
    - Maintains step tracking across interrupted training sessions
    - Progress bar synchronization with resumed step count
    
    ### Configuration Management
    - Comprehensive config extraction via utils.create_config()
    - Automatic model architecture documentation
    - Strategy hyperparameter logging
    - Training environment and hardware tracking
    
    ## Integration with ExoGym
    
    ### Strategy Callback Registration
    - Automatically registers with strategy LR callbacks
    - Receives real-time learning rate updates
    - Enables LR schedule visualization in WandB interface
    
    ### Perplexity Calculation
    - Automatic perplexity computation (exp(loss)) for language models
    - Logged alongside raw loss values for language modeling tasks
    - Available for both training and validation metrics
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_steps: int,
        strategy=None,
        train_node=None,
        wandb_project: str = None,
        run_name: str = None,
    ):
        """
        Initialize WandB logger with comprehensive configuration tracking.
        
        Args:
            model: PyTorch model being trained
            max_steps: Total training steps
            strategy: Training strategy for callback registration
            train_node: Training node for configuration extraction
            wandb_project: WandB project name for organization
            run_name: Custom run name (None for WandB auto-generation)
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Please install it using `pip install wandb`."
            )

        super().__init__(model, max_steps)

        self.wandb_project = wandb_project
        self.run_name = run_name or None

        # Extract comprehensive configuration for experiment tracking
        wandb_config = create_config(
            model=model,
            strategy=strategy,
            train_node=train_node,
            extra_config={
                "max_steps": max_steps,
            },
        )

        print(
            f'initialized wandb project with model size {wandb_config["model_parameters"]}'
        )

        # Configure WandB initialization with resume capability
        init_kwargs = {
            "project": self.wandb_project,
            "name": self.run_name,
            "config": wandb_config,
            "resume": "allow",  # Allow resuming if possible, or create new
        }

        wandb.init(**init_kwargs)

        # Synchronize step tracking with resumed WandB run
        print(
            f"Started new wandb run '{self.run_name}' (ID: {wandb.run.id}). Starting at step {self.step}."
        )

        # Update progress bar to match resumed step count
        self.pbar.n = self.step
        self.pbar.last_print_n = self.step
        self.pbar.refresh()

        # Register for learning rate callbacks from strategy
        strategy.lr_callbacks.append(self.log_lr)

    def log_loss(self, loss: float, name: str):
        """
        Log validation loss with automatic perplexity calculation to WandB.
        
        Uploads both raw loss value and computed perplexity to WandB cloud.
        The name parameter enables distinction between local and global
        model evaluation results in distributed training.
        
        Args:
            loss: Validation loss value
            name: Loss identifier ("local" or "global")
        """
        import wandb

        if hasattr(self, "run_name"):
            data = {f"{name}_loss": loss, f"{name}_perplexity": float(np.exp(loss))}
            wandb.log(data, step=self.step)

    def log_train(self, loss: float):
        """
        Log training loss with perplexity and LR to WandB and update progress bar.
        
        Combines WandB cloud logging with local progress bar updates for
        comprehensive training monitoring. Automatically computes perplexity
        and includes current learning rate if available.
        
        Args:
            loss: Current training loss value
        """
        import wandb

        # Log to WandB cloud if properly initialized
        if hasattr(self, "run_name"):
            data = {
                "train_loss": loss,
                "train_perplexity": float(np.exp(loss)),
            }
            if self.current_lr:
                data["lr"] = self.current_lr

            wandb.log(data, step=self.step)

        # Update local progress bar for immediate feedback
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )


class CSVLogger(Logger):
    """
    Local CSV file logger for offline training monitoring and CI/CD integration.
    
    CSVLogger provides lightweight, self-contained logging to local CSV files
    with comprehensive configuration persistence. Ideal for development environments,
    continuous integration pipelines, and scenarios where cloud logging is not
    available or desired.
    
    ## Key Features
    
    ### File Organization
    - Structured directory layout with run-specific folders
    - Separate CSV files for training and validation metrics
    - Complete configuration persistence in JSON format
    - Timestamp-based automatic run naming
    
    ### Self-Contained Logging
    - No external dependencies beyond Python standard library
    - Works offline without internet connectivity
    - Compatible with version control for experiment tracking
    - Easy integration with data analysis tools (pandas, Excel)
    
    ### Distributed Training Support
    - Separate columns for local vs global model evaluation
    - Handles rank-specific logging responsibilities
    - Coordinates with distributed evaluation patterns
    
    ## File Structure
    
    ```
    logs/
    └── {run_name}/
        ├── config.json          # Complete training configuration
        ├── train.csv           # Training metrics (loss, perplexity, LR)
        └── validation.csv      # Validation metrics (local/global)
    ```
    
    ## CSV Schema
    
    ### train.csv
    - step: Training step number
    - train_loss: Raw training loss value
    - train_perplexity: Computed perplexity (exp(loss))
    - lr: Current learning rate from scheduler
    
    ### validation.csv
    - step: Training step when validation was performed
    - local_loss/local_perplexity: Local model evaluation results
    - global_loss/global_perplexity: Global model evaluation results
    
    ## Configuration Persistence
    
    ### Comprehensive Config Extraction
    - Uses utils.create_config() for complete training documentation
    - Includes model architecture, strategy hyperparameters, and training settings
    - Enables full experiment reproducibility
    
    ### JSON Format
    - Human-readable configuration format
    - Compatible with analysis tools and scripts
    - Version control friendly for experiment tracking
    
    ## Usage Patterns
    
    ### Development and Debugging
    ```python
    logger = CSVLogger(
        model=model,
        max_steps=1000,
        strategy=strategy,
        train_node=train_node,
        log_dir="debug_logs"
    )
    ```
    
    ### CI/CD Integration
    ```python
    # Automatic timestamped runs for CI
    logger = CSVLogger(
        model=model,
        max_steps=1000,
        strategy=strategy,
        train_node=train_node
        # Auto-generated run name with timestamp
    )
    ```
    
    This logger provides reliable local logging without external dependencies,
    making it ideal for development, testing, and offline analysis workflows.
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_steps: int,
        strategy,
        train_node=None,
        log_dir: str = "logs",
        run_name: str = None,
    ):
        """
        Initialize CSV logger with local file structure and configuration persistence.
        
        Args:
            model: PyTorch model being trained
            max_steps: Total training steps
            strategy: Training strategy for configuration extraction
            train_node: Training node for configuration extraction
            log_dir: Base directory for log storage (default: "logs")
            run_name: Custom run name (auto-generated timestamp if None)
        """
        super().__init__(model, max_steps)

        # Generate timestamp-based run name if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_name = run_name
        self.log_dir = log_dir

        # Create structured directory layout
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Define CSV file paths for different metric types
        self.train_csv_path = os.path.join(self.run_dir, "train.csv")
        self.val_csv_path = os.path.join(self.run_dir, "validation.csv")

        # Initialize CSV files with appropriate headers
        self._init_csv_file(
            self.train_csv_path, ["step", "train_loss", "train_perplexity", "lr"]
        )
        self._init_csv_file(
            self.val_csv_path,
            [
                "step",
                "local_loss",
                "local_perplexity",
                "global_loss",
                "global_perplexity",
            ],
        )

        # Extract and persist complete training configuration
        config = create_config(
            model=model,
            strategy=strategy,
            train_node=train_node,
            extra_config={
                "max_steps": max_steps,
                "run_name": run_name,
                "log_dir": log_dir,
            },
        )

        # Save configuration as JSON for reproducibility
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(
            f'CSV Logger initialized with model size {config["model_parameters"]}M parameters'
        )
        print(f"Logging to directory: {self.run_dir}")

        # Register for learning rate callbacks from strategy
        strategy.lr_callbacks.append(self.log_lr)

    def _init_csv_file(self, filepath: str, headers: list):
        """
        Initialize CSV file with headers if it doesn't exist.
        
        Creates new CSV file with specified column headers. Only creates
        the file if it doesn't already exist, enabling safe resumption
        of interrupted training runs.
        
        Args:
            filepath: Path to CSV file to create
            headers: List of column header names
        """
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _write_csv_row(self, filepath: str, data: dict):
        """
        Append a data row to existing CSV file.
        
        Writes data dictionary to CSV file using the existing header
        structure. Reads headers from file to ensure correct column
        ordering regardless of dictionary key order.
        
        Args:
            filepath: Path to existing CSV file
            data: Dictionary of column_name -> value mappings
        """
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="") as f:
            if file_exists:
                # Read the header to get field order
                with open(filepath, "r") as read_f:
                    reader = csv.reader(read_f)
                    headers = next(reader)

                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(data)

    def log_loss(self, loss: float, name: str):
        """Log validation loss to CSV"""
        # Read existing row for this step if it exists
        existing_data = {}
        if os.path.exists(self.val_csv_path):
            with open(self.val_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["step"]) == self.step:
                        existing_data = row
                        break

        # Create or update the data for this step
        data = {
            "step": self.step,
            "local_loss": existing_data.get("local_loss", ""),
            "local_perplexity": existing_data.get("local_perplexity", ""),
            "global_loss": existing_data.get("global_loss", ""),
            "global_perplexity": existing_data.get("global_perplexity", ""),
        }

        # Update with the new loss data
        data[f"{name}_loss"] = loss
        data[f"{name}_perplexity"] = float(np.exp(loss))

        # If this step already exists, we need to update it
        if existing_data:
            # Rewrite the entire file with updated data
            temp_rows = []
            with open(self.val_csv_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for row in reader:
                    if int(row["step"]) == self.step:
                        temp_rows.append(data)
                    else:
                        temp_rows.append(row)

            with open(self.val_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(temp_rows)
        else:
            # Append new row
            self._write_csv_row(self.val_csv_path, data)

    def log_train(self, loss: float):
        """Log training loss to CSV"""
        data = {
            "step": self.step,
            "train_loss": loss,
            "train_perplexity": float(np.exp(loss)),
        }
        if self.current_lr:
            data["lr"] = self.current_lr

        self._write_csv_row(self.train_csv_path, data)

        # Update progress bar
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )
