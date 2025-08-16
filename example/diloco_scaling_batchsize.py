"""
Distributed Learning Coordination (DiLoCo) Batch Size Scaling Experiment

This file implements a comprehensive experiment comparing DiLoCo training strategy
against standard DDP (Distributed Data Parallel) across different batch sizes.
The experiment is designed to validate DiLoCo's scaling properties and measure
performance across various distributed configurations.

Role in System:
- Serves as a research experiment script for distributed training strategy comparison
- Tests DiLoCo's effectiveness vs standard DDP at different scales
- Generates comparative performance data for distributed training strategies

Called by:
- Researchers running batch size scaling experiments
- Automated experiment runners or CI systems
- Manual execution for distributed training analysis

Calls:
- exogym.trainer.LocalTrainer for distributed training coordination
- exogym.strategy.diloco.DiLoCoStrategy for DiLoCo distributed optimization
- exogym.strategy.strategy.SimpleReduceStrategy for standard DDP comparison
- nanogpt module for GPT model implementation and dataset loading

Experiment Structure:
1. Loads OpenWebText dataset with configurable data percentage per node
2. Creates a standard 50M parameter GPT model configuration
3. Tests multiple batch size multipliers (1x, 2x, 4x, 8x base size)
4. For each batch size, runs both DDP and DiLoCo with different node counts
5. Logs all results to WandB for performance comparison analysis

Data Flow:
- Dataset loading → Model creation → Strategy comparison → Results logging
- Each strategy run is independent to ensure fair comparison
- Results include loss curves, throughput metrics, and convergence analysis
"""

from exogym.trainer import LocalTrainer
from exogym.strategy.optim import OptimSpec
from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.strategy import SimpleReduceStrategy

from nanogpt import GPT, GPTConfig, get_dataset

import torch


class ExperimentConfig:
    """
    Distributed training experiment configuration constants.
    
    This class centralizes all experimental parameters to ensure consistency
    across training runs and make modifications clear for future researchers.
    Values are chosen based on distributed training best practices and
    computational resource constraints.
    """
    
    # ── Distributed Training Configuration ──────────────────────────────────
    
    MAX_NODES = 4
    """Maximum number of distributed nodes to test in experiments.
    
    Chosen as 4 to balance:
    - Meaningful distributed training validation (>1 node)
    - Computational resource constraints for research environments
    - DiLoCo paper validation which tested up to 8 nodes
    
    Supports testing K ∈ {1, 2, 4} node configurations for scaling analysis.
    """
    
    DILOCO_COMMUNICATION_INTERVAL = 30
    """DiLoCo communication interval (H parameter) in training steps.
    
    Represents number of local SGD steps between parameter synchronization.
    Value of 30 chosen based on:
    - DiLoCo paper recommendations for language modeling (H=100-200 for large models)
    - Scaled down for smaller model and batch sizes in this experiment
    - Balance between communication efficiency and convergence speed
    
    Higher H values reduce communication overhead but may hurt convergence.
    Lower H values increase communication but approach standard DDP behavior.
    """
    
    # ── Dataset and Sequence Configuration ──────────────────────────────────
    
    TOTAL_TOKENS = (2**15) * (2**13)  # 268,435,456 tokens total
    """Total number of tokens to process across all experiments.
    
    Calculation: 2^15 * 2^13 = 32,768 * 8,192 = 268M tokens
    
    This ensures consistent training volume across different batch sizes:
    - For smallest batch size: ~1024 training steps
    - For largest batch size: ~256 training steps
    - Provides sufficient data for meaningful loss comparison
    
    Alternative smaller configuration (commented): 2^15 * 10 = 327,680 tokens
    """
    
    SEQUENCE_LENGTH = 2**10  # 1024 tokens per sequence
    """Sequence length for GPT training blocks.
    
    Standard GPT-2 block size of 1024 tokens chosen for:
    - Compatibility with pretrained tokenizer vocabularies
    - Reasonable memory usage during distributed training
    - Consistency with OpenWebText preprocessing pipeline
    - Balance between context length and computational efficiency
    """
    
    BASE_BATCH_SIZE = 2**16  # 65,536 tokens per batch
    """Base global batch size in tokens for scaling experiments.
    
    Chosen as 65,536 tokens (64K) to:
    - Provide meaningful gradient estimates for stable training
    - Allow clean division by SEQUENCE_LENGTH (64K / 1K = 64 sequences)
    - Support multiple batch size scaling factors (1x, 2x, 4x, 8x)
    - Fit within memory constraints when distributed across nodes
    
    This translates to 64 sequences of 1024 tokens each at 1x multiplier.
    """
    
    # ── Experiment Scaling Configuration ────────────────────────────────────
    
    BATCH_SIZE_MULTIPLIERS = [1, 2, 4, 8]
    """Batch size scaling factors to test in experiments.
    
    Tests global batch sizes of:
    - 1x: 65,536 tokens (baseline)
    - 2x: 131,072 tokens (2x scaling)
    - 4x: 262,144 tokens (4x scaling) 
    - 8x: 524,288 tokens (8x scaling)
    
    Chosen to evaluate:
    - Linear scaling properties of DiLoCo vs DDP
    - Memory and computational scaling limits
    - Loss convergence at different batch sizes
    - Learning rate scaling requirements (lr ∝ batch_size)
    """
    
    NODE_CONFIGURATIONS = [1, 2, 4]
    """Number of nodes to test for each DiLoCo experiment.
    
    K=1: Single node baseline for comparison with DDP
    K=2: Minimal distributed setup to verify DiLoCo functionality  
    K=4: Full distributed setup using MAX_NODES for scaling analysis
    
    Each configuration uses identical total batch size, distributed across K nodes.
    """
    
    # ── Training Hyperparameters ────────────────────────────────────────────
    
    BASE_LEARNING_RATE = 0.001
    """Base learning rate for AdamW optimizer.
    
    Conservative learning rate of 1e-3 chosen for:
    - Stable training across different batch sizes and node configurations
    - Compatibility with learning rate scaling (lr *= batch_size_multiplier)
    - Good convergence properties for 50M parameter GPT models
    - Robustness to distributed training communication delays
    """
    
    GRADIENT_CLIP_NORM = 1.0
    """Maximum gradient norm for gradient clipping.
    
    Standard value of 1.0 chosen to:
    - Prevent gradient explosion during distributed training
    - Maintain training stability across different communication intervals
    - Follow GPT training best practices and literature recommendations
    - Ensure consistent optimization dynamics across node configurations
    """
    
    # ── Communication and Scheduling Configuration ──────────────────────────
    
    WARMUP_STEPS_BASE = 1024
    """Base number of warmup steps for learning rate scheduling.
    
    Scaled by batch_size_multiplier to maintain consistent warmup duration:
    - 1x batch: 1024 warmup steps
    - 2x batch: 512 warmup steps (half the steps, same token count)
    - 4x batch: 256 warmup steps
    - 8x batch: 128 warmup steps
    
    Ensures consistent warmup token count across all batch size configurations.
    """
    
    VALIDATION_INTERVAL_BASE = 100
    """Base validation interval for loss evaluation and logging.
    
    Scaled by batch_size_multiplier to maintain consistent validation frequency:
    - Larger batches validate less frequently (fewer steps needed)
    - Maintains approximately consistent token intervals for validation
    - Balances monitoring overhead with training efficiency
    """
    
    # ── Model Architecture Configuration ────────────────────────────────────
    
    GPT_LAYERS = 8
    """Number of transformer layers in the GPT model.
    
    Moderate size chosen for:
    - Sufficient model complexity to demonstrate distributed training benefits
    - Reasonable training time for batch size scaling experiments
    - Memory efficiency for multi-node distributed training
    - Good balance between underfitting and computational cost
    """
    
    GPT_HEADS = 8
    """Number of attention heads per transformer layer.
    
    Standard choice that:
    - Provides sufficient attention capacity for language modeling
    - Maintains clean embedding dimension factorization (512 / 8 = 64)
    - Balances model expressiveness with computational efficiency
    - Follows established transformer architecture patterns
    """
    
    GPT_EMBEDDING_DIM = 512
    """Model embedding dimension (d_model).
    
    Moderate size chosen for:
    - Reasonable parameter count (~50M parameters total)
    - Good representational capacity for language modeling
    - Efficient distributed training with multiple nodes
    - Balance between model capacity and training speed
    """

def main():
    """
    Execute distributed training experiment comparing DiLoCo vs DDP across batch sizes.
    
    This function implements the complete experimental pipeline:
    1. Dataset loading with distributed node configuration
    2. Model architecture setup using predefined configuration
    3. Training loop across all batch size multipliers
    4. Strategy comparison: DDP baseline followed by DiLoCo variants
    5. Results logging to WandB for analysis and visualization
    
    The experiment systematically evaluates:
    - Batch size scaling properties (1x, 2x, 4x, 8x)
    - Node count scaling for DiLoCo (K=1, 2, 4)
    - Learning rate scaling with batch size
    - Communication interval optimization
    
    Results provide comprehensive comparison between traditional DDP
    and DiLoCo distributed training across different scales.
    """
    # Dataset Loading with Distributed Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Load OpenWebText dataset with data allocation per node
    # Uses 0.5% of dataset per node to ensure sufficient training data
    # while maintaining reasonable experiment duration
    train_dataset, vocab_size = get_dataset(
        "owt",
        block_size=ExperimentConfig.SEQUENCE_LENGTH,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * ExperimentConfig.MAX_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        "owt", 
        block_size=ExperimentConfig.SEQUENCE_LENGTH, 
        device="cpu", 
        start_pc=0.99, 
        end_pc=1.0
    )
    # Alternative smaller dataset configurations for quick testing:
    # train_dataset, vocab_size = get_dataset(
    #     "shakespeare",
    #     block_size=ExperimentConfig.SEQUENCE_LENGTH,
    #     device="cpu",
    #     start_pc=0.0,
    #     end_pc=0.9
    # )
    # val_dataset, vocab_size = get_dataset(
    #     "shakespeare", 
    #     block_size=ExperimentConfig.SEQUENCE_LENGTH, 
    #     device="cpu", 
    #     start_pc=0.9,
    #     end_pc=1.0
    # )

    # Model Architecture Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Create GPT model with experimental configuration
    # Architecture designed for distributed training efficiency while
    # maintaining sufficient capacity for meaningful language modeling
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=ExperimentConfig.SEQUENCE_LENGTH,
        n_layer=ExperimentConfig.GPT_LAYERS,
        n_head=ExperimentConfig.GPT_HEADS,
        n_embd=ExperimentConfig.GPT_EMBEDDING_DIM,
        dropout=0.0,  # No dropout for cleaner experimental comparison
    )
    
    model = GPT(gpt_config)
    
    # Distributed Training Coordinator Setup
    # ─────────────────────────────────────────────────────────────────────────
    # LocalTrainer manages multi-node coordination and communication
    # start_port=12355 ensures no conflicts with other distributed training jobs
    trainer = LocalTrainer(
        model,
        train_dataset,
        val_dataset,
        start_port=12355
    )

    # Batch Size Scaling Experiment Loop
    # ─────────────────────────────────────────────────────────────────────────
    # Test each batch size multiplier with both DDP and DiLoCo strategies
    # Learning rate scales linearly with batch size following best practices
    for batch_size_multiplier in ExperimentConfig.BATCH_SIZE_MULTIPLIERS:
        # Calculate scaled training parameters for this batch size
        global_batch_tokens = batch_size_multiplier * ExperimentConfig.BASE_BATCH_SIZE
        scaled_learning_rate = ExperimentConfig.BASE_LEARNING_RATE * batch_size_multiplier
        scaled_warmup_steps = ExperimentConfig.WARMUP_STEPS_BASE // batch_size_multiplier
        scaled_val_interval = ExperimentConfig.VALIDATION_INTERVAL_BASE
        
        # DDP Baseline Strategy Configuration
        # ─────────────────────────────────────────────────────────────────────
        # Standard distributed data parallel training for comparison baseline
        # Uses simple gradient averaging across nodes with no communication optimization
        ddp_strategy = SimpleReduceStrategy(
            optim_spec=OptimSpec(
                torch.optim.AdamW, 
                lr=scaled_learning_rate,
            ),
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={
                "warmup_steps": scaled_warmup_steps,
                "cosine_anneal": True,
            },
            max_norm=ExperimentConfig.GRADIENT_CLIP_NORM,
        )

        # DDP Baseline Training Run
        # ─────────────────────────────────────────────────────────────────────
        # Single-node DDP baseline to establish performance comparison point
        trainer.fit(
            num_epochs=1,
            max_steps=ExperimentConfig.TOTAL_TOKENS // global_batch_tokens,
            strategy=ddp_strategy,
            num_nodes=1,
            device="mps",  # Apple Silicon MPS for local development
            batch_size=global_batch_tokens // ExperimentConfig.SEQUENCE_LENGTH,
            shuffle=True,
            val_size=512,
            val_interval=scaled_val_interval,
            wandb_project="DiLoCo-Batchsize-Scaling",
            run_name=f"ddp-batchsize{global_batch_tokens}",
        )

        # DiLoCo Multi-Node Scaling Experiments
        # ─────────────────────────────────────────────────────────────────────
        # Test DiLoCo across different node counts for scaling analysis
        # Each configuration maintains same total batch size, distributed across K nodes
        for K in ExperimentConfig.NODE_CONFIGURATIONS:
            diloco_strategy = DiLoCoStrategy(
                optim_spec=OptimSpec(
                    torch.optim.AdamW, 
                    lr=scaled_learning_rate
                ),
                lr_scheduler="lambda_cosine",
                lr_scheduler_kwargs={
                    "warmup_steps": scaled_warmup_steps,
                    "cosine_anneal": True,
                },
                max_norm=ExperimentConfig.GRADIENT_CLIP_NORM,
                H=ExperimentConfig.DILOCO_COMMUNICATION_INTERVAL,
            )

            # DiLoCo Distributed Training Execution
            # ─────────────────────────────────────────────────────────────────
            # Distributed training with communication interval H=30
            # Batch size automatically sharded across K nodes
            trainer.fit(
                num_epochs=1,
                max_steps=ExperimentConfig.TOTAL_TOKENS // global_batch_tokens,
                strategy=diloco_strategy,
                num_nodes=K,
                device="mps",
                batch_size=global_batch_tokens // ExperimentConfig.SEQUENCE_LENGTH // K,
                minibatch_size=32 // K,  # Conservative gradient accumulation for memory efficiency
                shuffle=True,
                val_size=256,
                val_interval=max(1, scaled_val_interval // batch_size_multiplier),
                wandb_project="DiLoCo-Batchsize-Scaling",
                run_name=f"diloco-K{K}-batchsize{global_batch_tokens}",
            )


if __name__ == "__main__":
    main()
