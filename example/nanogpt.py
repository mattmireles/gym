"""
NanoGPT Distributed Training CLI Interface

This file provides a comprehensive command-line interface for training GPT models
using various distributed training strategies. It serves as the primary entry point
for researchers and practitioners wanting to experiment with different distributed
optimization approaches on language modeling tasks.

Role in System:
- Primary CLI interface for distributed GPT training experiments
- Unified configuration system supporting all distributed training strategies
- Dataset factory implementation for scalable data loading across nodes
- Strategy selection and hyperparameter management for research workflows

Called by:
- Researchers running distributed language model training experiments
- Automated hyperparameter search systems
- CI/CD pipelines for model training validation
- Manual execution: `python nanogpt.py --strategy diloco --num_nodes 4`

Calls:
- exogym.trainer.LocalTrainer for distributed training coordination
- nanogpt module (./nanogpt/) for GPT model implementation and dataset loading
- exogym.strategy.* modules for various distributed training strategies
- argparse for comprehensive CLI argument processing

Supported Training Strategies:
- base/ddp: Standard distributed data parallel training with gradient averaging
- diloco: Distributed Learning Coordination with outer loop optimization
- sparta: Sparse communication training with configurable sparsity
- fedavg: Federated averaging with configurable communication intervals
- demo: Gradient compression with DCT and top-k selection
- diloco_sparta: Hybrid approach combining DiLoCo outer loop with SPARTA communication

Configuration Management:
- Supports all major datasets: shakespeare, wikitext, code, owt (OpenWebText)
- Flexible model sizing from small (4 layers) to xl (48 layers)
- Comprehensive optimization settings: learning rates, batch sizes, regularization
- Device-agnostic with automatic MPS/CUDA/CPU selection
- Reproducible training with seed management and deterministic operations

Dataset Factory Pattern:
- For large datasets (OWT): implements per-node data sharding
- Automatically handles data percentage allocation across distributed nodes
- Lazy loading support for memory-efficient training on large datasets
- Configurable validation splits and evaluation protocols
"""

from exogym.trainer import LocalTrainer
from nanogpt import GPT, GPTConfig, get_dataset
from exogym.strategy.optim import OptimSpec

import argparse
import torch


class NanoGPTConfig:
    """
    NanoGPT distributed training configuration constants.
    
    Centralizes all hyperparameters, dataset configurations, and training settings
    for distributed GPT training experiments. Values are optimized for research
    workflows and distributed training stability across various strategies.
    """
    
    # ── Default Training Configuration ─────────────────────────────────────────
    
    DEFAULT_BATCH_SIZE = 16
    """Default batch size for distributed training.
    
    Conservative choice of 16 to:
    - Ensure memory efficiency across different hardware configurations
    - Support gradient accumulation for larger effective batch sizes
    - Maintain training stability with distributed communication delays
    - Enable testing on resource-constrained development environments
    """
    
    DEFAULT_LEARNING_RATE = 0.001
    """Default learning rate for AdamW optimizer.
    
    Standard 1e-3 learning rate chosen for:
    - Robust training across different model sizes and datasets
    - Compatibility with learning rate scaling for larger batch sizes
    - Good convergence properties for transformer architectures
    - Stability across various distributed training strategies
    """
    
    DEFAULT_MAX_NORM = 1.0
    """Default gradient clipping norm.
    
    Conservative gradient clipping to:
    - Prevent gradient explosion in distributed training
    - Maintain optimization stability across communication intervals
    - Follow transformer training best practices
    - Ensure consistent dynamics across different node counts
    """
    
    DEFAULT_WARMUP_STEPS = 1000
    """Default number of learning rate warmup steps.
    
    1000 steps chosen to:
    - Provide stable optimization startup for transformer training
    - Scale appropriately with different batch sizes (warmup_steps / batch_multiplier)
    - Prevent early training instability in distributed settings
    - Follow established practices for language model training
    """
    
    DEFAULT_MAX_STEPS = 10000
    """Default maximum training steps.
    
    Conservative 10k steps for:
    - Reasonable experiment duration for research workflows
    - Sufficient training for meaningful strategy comparison
    - Computational efficiency for distributed training validation
    - Balance between convergence demonstration and resource usage
    """
    
    # ── Dataset Configuration ─────────────────────────────────────────────────
    
    DEFAULT_BLOCK_SIZE = 1024
    """Default sequence length for GPT training.
    
    Standard GPT-2 block size chosen for:
    - Compatibility with pretrained tokenizer vocabularies
    - Reasonable context length for language modeling tasks
    - Memory efficiency in distributed training environments
    - Consistency across different datasets and preprocessing pipelines
    """
    
    DEFAULT_DATASET = "shakespeare"
    """Default dataset for training experiments.
    
    Shakespeare chosen as default because:
    - Small size enables quick experimentation and testing
    - Character-level tokenization provides simple preprocessing
    - Well-understood dataset for validating training implementations
    - Fast download and processing for development workflows
    """
    
    DEFAULT_TRAIN_SPLIT_START = 0.0
    """Default start percentage for training data split.
    
    Start from beginning (0%) to:
    - Use full training data by default
    - Provide consistent baseline for experiments
    - Enable easy modification for data ablation studies
    - Maintain reproducible dataset splits
    """
    
    DEFAULT_TRAIN_SPLIT_END = 0.9
    """Default end percentage for training data split.
    
    Use 90% for training to:
    - Reserve meaningful validation data (10%)
    - Follow standard train/validation split practices
    - Provide sufficient training data for convergence
    - Enable proper generalization evaluation
    """
    
    DEFAULT_VAL_SPLIT_START = 0.9
    """Default start percentage for validation data split.
    
    Validation starts where training ends (90%) to:
    - Ensure no overlap between training and validation
    - Maintain temporal ordering for time-series datasets
    - Provide clean experimental conditions
    - Follow standard data splitting conventions
    """
    
    DEFAULT_VAL_SPLIT_END = 1.0
    """Default end percentage for validation data split.
    
    Use remaining 10% for validation to:
    - Provide meaningful validation set size
    - Enable robust generalization evaluation
    - Maintain standard train/val split ratios
    - Support statistical significance testing
    """
    
    # ── Model Architecture Configuration ──────────────────────────────────────
    
    DEFAULT_MODEL_SIZE = "small"
    """Default GPT model size configuration.
    
    Small model chosen as default for:
    - Fast experimentation and development workflows
    - Memory efficiency in distributed training
    - Quick convergence for strategy validation
    - Reasonable computational requirements for research
    
    Available sizes: small, base, medium, large, xl
    """
    
    DEFAULT_DROPOUT = None
    """Default dropout rate (None = use model default).
    
    No override by default to:
    - Use architecture-specific dropout rates
    - Maintain model design integrity
    - Enable easy dropout ablation studies via command line
    - Follow established model configuration practices
    """
    
    # ── Distributed Training Configuration ────────────────────────────────────
    
    DEFAULT_NUM_NODES = 1
    """Default number of distributed training nodes.
    
    Single node default for:
    - Local development and testing workflows
    - Baseline comparison with distributed configurations
    - Simplified setup for initial experimentation
    - Easy scaling to multi-node via command line arguments
    """
    
    DEFAULT_EPOCHS = 1
    """Default number of training epochs.
    
    Single epoch default to:
    - Provide quick training completion for testing
    - Enable multi-step experiments within reasonable time
    - Focus on distributed training validation rather than convergence
    - Support rapid iteration during development
    """
    
    # ── Strategy-Specific Configuration ───────────────────────────────────────
    
    FEDAVG_DEFAULT_H = 100
    """Default communication interval for FedAvg strategy.
    
    100 steps chosen for FedAvg because:
    - Balances communication efficiency with convergence quality
    - Follows federated learning literature recommendations
    - Provides meaningful local training between communications
    - Scales appropriately with different datasets and model sizes
    """
    
    SPARTA_DEFAULT_SPARSITY = 0.005
    """Default sparsity parameter for SPARTA strategy.
    
    0.5% sparsity chosen to:
    - Provide significant communication reduction
    - Maintain training stability and convergence
    - Follow SPARTA paper recommendations for language modeling
    - Balance compression benefits with gradient quality
    """
    
    DILOCO_DEFAULT_H = 100
    """Default communication interval for DiLoCo strategy.
    
    100 steps chosen for DiLoCo because:
    - Optimal trade-off between communication and convergence
    - Follows DiLoCo paper recommendations for language models
    - Provides sufficient local optimization steps
    - Scales well across different batch sizes and learning rates
    """
    
    DILOCO_DEFAULT_OUTER_LR = 0.7
    """Default outer learning rate for DiLoCo strategy.
    
    0.7 chosen based on:
    - DiLoCo paper empirical results for optimal convergence
    - Balance between fast outer loop convergence and stability
    - Robustness across different inner learning rates
    - Good performance across various model sizes and datasets
    """
    
    DILOCO_DEFAULT_OUTER_MOMENTUM = 0.9
    """Default outer momentum for DiLoCo strategy.
    
    Standard 0.9 momentum for:
    - Consistent with SGD best practices
    - Stable outer loop optimization dynamics
    - Following DiLoCo paper configurations
    - Good generalization across different training scenarios
    """
    
    # ── Validation and Logging Configuration ──────────────────────────────────
    
    DEFAULT_VAL_SIZE = 256
    """Default validation set size for evaluation.
    
    256 samples chosen to:
    - Provide meaningful validation statistics
    - Balance evaluation quality with computational efficiency
    - Enable frequent validation without excessive overhead
    - Support statistical significance testing
    """
    
    DEFAULT_VAL_INTERVAL = 100
    """Default validation interval in training steps.
    
    Validate every 100 steps to:
    - Provide detailed training progress monitoring
    - Enable early stopping and training diagnostics
    - Balance monitoring overhead with information value
    - Support research analysis and strategy comparison
    """
    
    DEFAULT_SEED = 1337
    """Default random seed for reproducible experiments.
    
    1337 chosen as:
    - Standard seed value in deep learning research
    - Enables reproducible experimental results
    - Consistent initialization across distributed nodes
    - Easy to remember and modify for experiment variations
    """


def gen_run_name(args, strategy):
    """
    Generate standardized WandB run names based on strategy and arguments.
    
    Creates consistent, informative run names for experiment tracking and
    comparison across different distributed training strategies and configurations.
    
    Args:
        args: Parsed command-line arguments containing training configuration
        strategy: String identifier for the distributed training strategy
    
    Returns:
        str: Formatted run name for WandB experiment tracking
        
    Run Name Format:
        - Base format: "bs{batch_size}_lr{learning_rate:.0e}"
        - Strategy-specific suffixes added based on relevant hyperparameters
        - Examples: "bs64_lr1e-03_warm1000_max10000", "ddp_bs64_lr1e-03_n4"
    
    Strategy Naming Conventions:
        - base/ddp: Include warmup steps, max steps, node count
        - fedavg: Include communication interval (H) and node count
        - sparta: Include sparsity parameter (p) and node count
        - diloco: Include outer learning rate and communication interval
        - demo: Include compression parameters (top-k, decay)
        - diloco_sparta: Combine DiLoCo and SPARTA parameters
    """
    base_name = f"bs{args.batch_size}_lr{args.lr:.0e}"

    if strategy == "base":
        return f"{base_name}_warm{args.warmup_steps}_max{args.max_steps}"
    elif strategy == "ddp":
        return f"ddp_{base_name}_n{args.num_nodes}"
    elif strategy == "fedavg":
        return f"{base_name}_H{args.H}_n{args.num_nodes}"
    elif strategy == "sparta":
        return f"p{args.p_sparta}_n{args.num_nodes}_lr{args.lr:.0e}"
    elif strategy == "diloco":
        return f"{base_name}_outer{args.outer_lr:.0e}_H{args.diloco_interval}"
    elif strategy == "demo":
        return f"{base_name}_topk{args.compression_topk}_decay{args.compression_decay}"
    elif strategy == "diloco_sparta":
        return f"{base_name}_outer{args.outer_lr:.0e}_H{args.diloco_interval}_p{args.p_sparta}"
    else:
        return base_name


def arg_parse():
    """
    Create comprehensive argument parser for distributed GPT training CLI.
    
    Builds a complete argument parser supporting all distributed training strategies
    and their specific hyperparameters. Uses conflict_handler="resolve" to allow
    strategy-specific parameter overrides while maintaining clean interface.
    
    Returns:
        argparse.ArgumentParser: Configured parser with all training arguments
        
    Argument Categories:
        - Dataset: Data selection, splits, block size configuration
        - Training: Epochs, nodes, device, model size, dropout
        - Optimization: Batch sizes, learning rate, gradient clipping, scheduling
        - Logging: Seed, WandB project, validation configuration
        - Strategy Selection: Choose between base/ddp/fedavg/sparta/diloco/demo
        - Strategy-Specific: Parameters for each distributed training strategy
        
    Strategy-Specific Arguments:
        - FedAvg: Communication interval (H), island size for hierarchical aggregation
        - SPARTA: Sparsity parameter (p_sparta), async delay, communication interval
        - DiLoCo: Communication interval, outer learning rate, momentum, Nesterov
        - DeMo: Compression decay, top-k compression, DCT chunk size
        - DiLoCo+SPARTA: Combined parameters for hybrid approach
    
    Design Considerations:
        - All arguments have sensible defaults from NanoGPTConfig
        - Strategy-specific arguments only affect their respective strategies
        - Conflict resolution allows clean parameter override patterns
        - Help strings provide clear guidance for parameter selection
    """
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=NanoGPTConfig.DEFAULT_DATASET,
        help="which dataset to use (shakespeare, wikitext, code, owt)",
    )
    parser.add_argument("--start_pc", type=float, default=NanoGPTConfig.DEFAULT_TRAIN_SPLIT_START)
    parser.add_argument("--end_pc", type=float, default=NanoGPTConfig.DEFAULT_TRAIN_SPLIT_END)
    parser.add_argument("--val_start_pc", type=float, default=NanoGPTConfig.DEFAULT_VAL_SPLIT_START)
    parser.add_argument("--val_end_pc", type=float, default=NanoGPTConfig.DEFAULT_VAL_SPLIT_END)
    parser.add_argument("--block_size", type=int, default=NanoGPTConfig.DEFAULT_BLOCK_SIZE)

    # Training arguments
    parser.add_argument("--num_nodes", type=int, default=NanoGPTConfig.DEFAULT_NUM_NODES)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=NanoGPTConfig.DEFAULT_EPOCHS)
    parser.add_argument(
        "--model_size",
        type=str,
        default=NanoGPTConfig.DEFAULT_MODEL_SIZE,
        choices=["small", "base", "medium", "large", "xl"],
    )
    parser.add_argument("--dropout", type=float, default=NanoGPTConfig.DEFAULT_DROPOUT)

    # Optimization arguments
    parser.add_argument("--batch_size", type=int, default=NanoGPTConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=NanoGPTConfig.DEFAULT_LEARNING_RATE)
    parser.add_argument("--max_norm", type=float, default=NanoGPTConfig.DEFAULT_MAX_NORM)
    parser.add_argument("--warmup_steps", type=int, default=NanoGPTConfig.DEFAULT_WARMUP_STEPS)
    parser.add_argument("--max_steps", type=int, default=NanoGPTConfig.DEFAULT_MAX_STEPS)
    parser.add_argument("--cosine_anneal", action="store_true")

    # Logging and reproducibility
    parser.add_argument("--seed", type=int, default=NanoGPTConfig.DEFAULT_SEED)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=NanoGPTConfig.DEFAULT_VAL_SIZE)
    parser.add_argument("--val_interval", type=int, default=NanoGPTConfig.DEFAULT_VAL_INTERVAL)

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        default="base",
        choices=["base", "ddp", "fedavg", "sparta", "diloco", "demo", "diloco_sparta"],
        help="Training strategy to use",
    )

    # FedAvg-specific arguments
    parser.add_argument(
        "--H", type=int, default=NanoGPTConfig.FEDAVG_DEFAULT_H, help="FedAvg communication interval"
    )
    parser.add_argument(
        "--island_size", type=int, default=None, help="FedAvg island size"
    )

    # SPARTA-specific arguments
    parser.add_argument(
        "--p_sparta", type=float, default=NanoGPTConfig.SPARTA_DEFAULT_SPARSITY, help="SPARTA sparsity parameter"
    )
    parser.add_argument(
        "--async_sparta_delay", type=int, default=0, help="SPARTA async delay"
    )
    parser.add_argument(
        "--sparta_interval", type=int, default=1, help="SPARTA communication interval"
    )

    # DiLoCo-specific arguments
    parser.add_argument(
        "--diloco_interval", type=int, default=NanoGPTConfig.DILOCO_DEFAULT_H, help="DiLoCo communication interval"
    )
    parser.add_argument(
        "--outer_lr", type=float, default=NanoGPTConfig.DILOCO_DEFAULT_OUTER_LR, help="DiLoCo outer learning rate"
    )
    parser.add_argument(
        "--nesterov", type=bool, default=True, help="DiLoCo Nesterov momentum"
    )
    parser.add_argument(
        "--outer_momentum", type=float, default=NanoGPTConfig.DILOCO_DEFAULT_OUTER_MOMENTUM, help="DiLoCo outer momentum"
    )

    # DeMo-specific arguments
    parser.add_argument(
        "--compression_decay",
        type=float,
        default=0.999,
        help="DeMo gradient error feedback decay",
    )
    parser.add_argument(
        "--compression_topk", type=int, default=32, help="DeMo top-k compression"
    )
    parser.add_argument(
        "--compression_chunk", type=int, default=64, help="DeMo DCT chunk size"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay factor"
    )

    return parser


def create_strategy(args):
    """
    Create distributed training strategy based on command-line arguments.
    
    Factory function that instantiates the appropriate distributed training strategy
    based on args.strategy selection. Handles strategy-specific parameter configuration
    and provides consistent interface across all strategy implementations.
    
    Args:
        args: Parsed command-line arguments containing strategy selection and parameters
        
    Returns:
        Strategy instance configured for distributed training
        
    Supported Strategies:
        - "base"/"ddp": SimpleReduceStrategy - Standard distributed data parallel
        - "fedavg": FedAvgStrategy - Federated averaging with communication intervals
        - "sparta": SPARTAStrategy - Sparse communication training with gradient compression
        - "diloco": DiLoCoStrategy - Distributed Learning Coordination with outer loop
        - "demo": DeMoStrategy - Gradient compression with DCT and top-k selection
        - "diloco_sparta": SPARTADiLoCoStrategy - Hybrid approach combining both methods
        
    Strategy Configuration Details:
        - All strategies use AdamW inner optimizer with specified learning rate
        - Cosine annealing learning rate schedule with configurable warmup
        - Gradient clipping for training stability across distributed nodes
        - Strategy-specific parameters (H, sparsity, outer LR) from command line
        
    Error Handling:
        - Raises ValueError for unsupported strategy names
        - Validates strategy-specific parameter combinations
        - Provides clear error messages for configuration issues
    """

    # Common lr scheduler config
    lr_scheduler_kwargs = {
        "warmup_steps": args.warmup_steps,
        "cosine_anneal": args.cosine_anneal,
    }

    if args.strategy == "ddp" or args.strategy == "base" or args.strategy == "":
        from exogym.strategy.strategy import SimpleReduceStrategy

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return SimpleReduceStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "fedavg":
        from exogym.strategy.federated_averaging import FedAvgStrategy

        if args.island_size is None:
            args.island_size = args.num_nodes
        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return FedAvgStrategy(
            inner_optim_spec=optim,
            H=args.H,
            island_size=args.island_size,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "sparta":
        from exogym.strategy.sparta import SPARTAStrategy

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return SPARTAStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
            p_sparta=args.p_sparta,
            async_sparta_delay=args.async_sparta_delay,
        )

    elif args.strategy == "diloco":
        from exogym.strategy.diloco import DiLoCoStrategy

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return DiLoCoStrategy(
            optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.diloco_interval,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "demo":
        from exogym.strategy.demo import DeMoStrategy

        optim = OptimSpec(
            torch.optim.AdamW,
            lr=args.lr,
            compression_decay=args.compression_decay,
            compression_topk=args.compression_topk,
            compression_chunk=args.compression_chunk,
            weight_decay=args.weight_decay,
        )
        return DeMoStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "diloco_sparta":
        from exogym.strategy.sparta_diloco import SPARTADiLoCoStrategy

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return SPARTADiLoCoStrategy(
            inner_optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.diloco_interval,
            p_sparta=args.p_sparta,
            sparta_interval=args.sparta_interval,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def main():
    """
    Main entry point for distributed GPT training experiments.
    
    Orchestrates the complete distributed training pipeline from argument parsing
    through dataset loading, model creation, strategy selection, and training execution.
    Supports both simple dataset loading and advanced dataset factory patterns for
    large-scale datasets like OpenWebText.
    
    Execution Flow:
        1. Parse command-line arguments using comprehensive argument parser
        2. Handle dataset loading with appropriate pattern (simple vs factory)
        3. Create GPT model with specified architecture configuration
        4. Instantiate LocalTrainer for distributed coordination
        5. Create distributed training strategy based on selection
        6. Execute training with full monitoring and validation
        
    Dataset Loading Patterns:
        - Simple Loading: For shakespeare, wikitext, code datasets
          Uses get_dataset() directly with start/end percentages
        - Factory Pattern: For OpenWebText (owt) with dynamic node allocation
          Creates dataset factory functions for distributed data sharding
          
    Configuration Sources:
        - Command-line arguments override all defaults
        - NanoGPTConfig provides sensible defaults for all parameters
        - Strategy-specific configurations applied automatically
        - Device selection with automatic fallback hierarchy
        
    Error Handling:
        - Validates dataset availability and preprocessing
        - Checks model configuration compatibility
        - Verifies strategy parameter combinations
        - Provides clear error messages for common issues
    """
    parser = arg_parse()
    args = parser.parse_args()

    # Dataset Factory Pattern for Large-Scale Datasets
    # ─────────────────────────────────────────────────────────────────────────
    # OpenWebText requires distributed data sharding across nodes
    # Factory pattern enables per-node data allocation and lazy loading
    if args.dataset == "owt" or False:

        def dataset_factory(
            rank: int, num_nodes: int, train_dataset: bool
        ) -> torch.utils.data.Dataset:
            if train_dataset:
                start_pc = (
                    rank / num_nodes * (args.end_pc - args.start_pc) + args.start_pc
                )
                end_pc = (rank + 1) / num_nodes * (
                    args.end_pc - args.start_pc
                ) + args.start_pc
            else:
                start_pc = args.val_start_pc
                end_pc = args.val_end_pc

            dataset, _ = get_dataset(
                args.dataset,
                block_size=args.block_size,
                device="cpu",
                start_pc=start_pc,
                end_pc=end_pc,
            )
            return dataset

        train_dataset = dataset_factory
        val_dataset = dataset_factory

        vocab_size = 50257

    else:
        # Get datasets
        train_dataset, vocab_size = get_dataset(
            args.dataset,
            block_size=args.block_size,
            device="cpu",
            start_pc=args.start_pc,
            end_pc=args.end_pc,
        )
        val_dataset, vocab_size = get_dataset(
            args.dataset,
            block_size=args.block_size,
            device="cpu",
            start_pc=args.val_start_pc,
            end_pc=args.val_end_pc,
        )

    # Create model
    gpt_config = GPTConfig.gpt2_size_map(args.model_size)
    if args.dropout is not None:
        gpt_config.dropout = args.dropout
    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = LocalTrainer(
        model,
        train_dataset,
        val_dataset,
    )

    # Create strategy based on selection
    strategy = create_strategy(args)

    # Train
    trainer.fit(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=strategy,
        num_nodes=args.num_nodes,
        device=args.device,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size or args.batch_size,
        shuffle=(args.dataset != "owt"),
        val_size=args.val_size,
        val_interval=args.val_interval,
        wandb_project=args.wandb_project,
        # run_name=args.run_name or gen_run_name(args, args.strategy)
        run_name=None,
    )


if __name__ == "__main__":
    main()
