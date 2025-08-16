"""
DiLoCo Training Playground - Minimal Example

This file provides the simplest possible example of training a GPT model using
the DiLoCo (Distributed Learning Coordination) strategy. Designed as a starting
point for understanding distributed training concepts and as a template for
rapid experimentation.

Role in System:
- Educational entry point for understanding DiLoCo distributed training
- Minimal working example with clearly marked modification points
- Template for researchers to quickly set up distributed training experiments
- Reference implementation showing proper exogym usage patterns

Called by:
- New users learning distributed training concepts
- Researchers prototyping new distributed training ideas
- Educational demonstrations and tutorials
- Quick experimentation with DiLoCo hyperparameters

Calls:
- exogym.trainer.LocalTrainer for distributed training coordination
- exogym.strategy.diloco.DiLoCoStrategy for distributed optimization
- nanogpt module for GPT model implementation and OpenWebText dataset

Configuration Highlights:
- Uses OpenWebText dataset with 0.5% data per node (0.005 * NUM_NODES)
- Small but meaningful GPT model: 8 layers, 8 heads, 512 embedding dimension
- DiLoCo with H=200 communication interval for balance of efficiency and convergence
- Apple Silicon MPS support for local development and testing
- Conservative batch size (16) with gradient accumulation (minibatch_size=8)

Modification Points:
- NUM_NODES: Change distributed node count
- Strategy parameters (H, learning rate, optimizer settings)
- Model architecture (layers, heads, embedding dimensions)
- Dataset selection and percentage
- Training duration (max_steps, num_epochs)
"""

from exogym.trainer import LocalTrainer
from nanogpt import GPT, GPTConfig, get_dataset
from exogym.strategy.optim import OptimSpec
import torch


class PlaygroundConfig:
    """
    Playground experiment configuration constants.
    
    Minimal configuration for quick DiLoCo experimentation and learning.
    Values chosen for educational clarity and fast iteration cycles.
    """
    
    # ── Distributed Training Configuration ─────────────────────────────────────
    
    NUM_NODES = 4
    """Number of distributed training nodes for DiLoCo experiment.
    
    4 nodes chosen to:
    - Demonstrate meaningful distributed training effects
    - Show DiLoCo communication optimization benefits
    - Provide realistic multi-node training experience
    - Balance experimental complexity with learning objectives
    
    Alternative configurations: NUM_NODES = 2 for simpler setup
    """
    
    # ── Dataset Configuration ──────────────────────────────────────────────────
    
    DATASET_NAME = "owt"
    """Dataset selection for training experiment.
    
    OpenWebText chosen for:
    - Realistic language modeling task demonstration
    - Sufficient complexity to show distributed training benefits
    - Standard benchmark for GPT training validation
    - Good balance between dataset size and experiment duration
    """
    
    BLOCK_SIZE = 1024
    """Sequence length for GPT training blocks.
    
    Standard 1024 token sequences for:
    - Compatibility with GPT-2 tokenizer and preprocessing
    - Reasonable context length for language modeling
    - Memory efficiency in distributed training
    - Consistency with larger-scale training pipelines
    """
    
    DATA_PERCENTAGE_PER_NODE = 0.005
    """Percentage of dataset to use per distributed node.
    
    0.5% per node (2% total for 4 nodes) chosen to:
    - Provide sufficient training data for meaningful results
    - Keep experiment duration reasonable for educational purposes
    - Demonstrate dataset scaling with node count
    - Enable quick iteration and experimentation
    """
    
    VALIDATION_START_PERCENTAGE = 0.99
    """Start percentage for validation data split.
    
    Use final 1% of dataset for validation to:
    - Provide clean separation from training data
    - Enable meaningful generalization evaluation
    - Maintain temporal ordering for time-series datasets
    - Follow standard validation practices
    """
    
    VALIDATION_END_PERCENTAGE = 1.0
    """End percentage for validation data split.
    
    Complete final 1% for validation to:
    - Provide sufficient validation samples
    - Enable robust performance evaluation
    - Support training progress monitoring
    - Maintain statistical significance
    """
    
    # ── Model Architecture Configuration ───────────────────────────────────────
    
    GPT_LAYERS = 8
    """Number of transformer layers in GPT model.
    
    8 layers chosen for:
    - Meaningful model complexity without excessive computation
    - Fast training convergence for educational demonstration
    - Reasonable memory requirements for distributed training
    - Good balance between model capacity and training speed
    """
    
    GPT_HEADS = 8
    """Number of attention heads per transformer layer.
    
    8 heads provide:
    - Sufficient attention capacity for language modeling
    - Clean factorization with embedding dimension (512 / 8 = 64)
    - Standard choice following transformer architecture patterns
    - Good balance between expressiveness and efficiency
    """
    
    GPT_EMBEDDING_DIM = 512
    """Model embedding dimension (d_model).
    
    512 dimensions chosen for:
    - Moderate model size (~50M parameters total)
    - Efficient distributed training across multiple nodes
    - Reasonable computational requirements for experimentation
    - Good representational capacity for language modeling
    """
    
    DROPOUT_RATE = 0.0
    """Dropout rate for model regularization.
    
    No dropout (0.0) for:
    - Cleaner experimental results without regularization noise
    - Focus on distributed training effects rather than regularization
    - Simplified model behavior for educational demonstration
    - Faster convergence in short experiment duration
    """
    
    # ── Training Configuration ─────────────────────────────────────────────────
    
    NUM_EPOCHS = 1
    """Number of training epochs.
    
    Single epoch chosen for:
    - Quick experiment completion for rapid iteration
    - Focus on distributed training mechanics rather than convergence
    - Educational demonstration within reasonable time constraints
    - Sufficient duration to observe DiLoCo communication effects
    """
    
    MAX_STEPS = 5000
    """Maximum number of training steps.
    
    5000 steps provide:
    - Sufficient training for meaningful loss reduction
    - Reasonable experiment duration for educational purposes
    - Multiple communication intervals to demonstrate DiLoCo
    - Good balance between learning demonstration and time efficiency
    """
    
    # ── DiLoCo Strategy Configuration ──────────────────────────────────────────
    
    DILOCO_COMMUNICATION_INTERVAL = 200
    """DiLoCo communication interval (H parameter) in training steps.
    
    200 steps chosen for:
    - Optimal balance between communication efficiency and convergence
    - Sufficient local training between parameter synchronizations
    - Demonstration of DiLoCo's reduced communication requirements
    - Good performance based on DiLoCo paper recommendations
    
    Higher H values reduce communication overhead but may hurt convergence.
    Lower H values increase communication but approach standard DDP behavior.
    """
    
    LEARNING_RATE = 0.0004
    """Learning rate for AdamW inner optimizer.
    
    Conservative 4e-4 learning rate for:
    - Stable training across distributed nodes
    - Good convergence properties for GPT architectures
    - Robustness to communication delays in DiLoCo
    - Following established practices for transformer training
    """
    
    # ── Optimization Configuration ─────────────────────────────────────────────
    
    WARMUP_STEPS = 1000
    """Number of learning rate warmup steps.
    
    1000 steps chosen to:
    - Provide stable optimization startup for transformer training
    - Prevent early training instability in distributed settings
    - Follow established practices for language model training
    - Appropriate scale for 5000 total training steps
    """
    
    COSINE_ANNEALING = True
    """Enable cosine annealing learning rate schedule.
    
    Cosine annealing enabled for:
    - Smooth learning rate decay promoting stable convergence
    - Better final model performance compared to constant learning rate
    - Standard practice in transformer training
    - Good interaction with DiLoCo outer loop optimization
    """
    
    GRADIENT_CLIP_NORM = 1.0
    """Maximum gradient norm for gradient clipping.
    
    Standard 1.0 clipping for:
    - Preventing gradient explosion in distributed training
    - Maintaining training stability across communication intervals
    - Following transformer training best practices
    - Ensuring consistent optimization dynamics across nodes
    """
    
    # ── Batch Size and Memory Configuration ────────────────────────────────────
    
    BATCH_SIZE = 16
    """Global batch size in sequences.
    
    Conservative batch size of 16 for:
    - Memory efficiency across different hardware configurations
    - Stable gradient estimates for optimization
    - Efficient distributed training with gradient accumulation
    - Support for resource-constrained development environments
    """
    
    MINIBATCH_SIZE = 8
    """Minibatch size for gradient accumulation.
    
    Gradient accumulation (16 // 8 = 2 steps) for:
    - Memory efficiency on smaller GPUs and Apple Silicon MPS
    - Flexibility for different hardware configurations
    - Conservative memory usage for stable training
    - Good balance between memory efficiency and training speed
    """
    
    # ── Validation and Monitoring Configuration ────────────────────────────────
    
    VALIDATION_SIZE = 256
    """Number of validation samples for evaluation.
    
    256 samples chosen to:
    - Provide meaningful validation statistics
    - Balance evaluation quality with computational efficiency
    - Enable frequent validation monitoring
    - Support training progress analysis
    """
    
    VALIDATION_INTERVAL = 100
    """Validation evaluation interval in training steps.
    
    Validate every 100 steps to:
    - Provide detailed training progress monitoring
    - Enable early detection of training issues
    - Support research analysis and debugging
    - Balance monitoring frequency with computational overhead
    """
    
    # ── Device Configuration ───────────────────────────────────────────────────
    
    DEVICE = "mps"
    """Target device for training computation.
    
    Apple Silicon MPS chosen for:
    - Local development and experimentation on Mac hardware
    - Good performance for educational and research workflows
    - Accessibility for developers without dedicated GPU clusters
    - Reasonable training speed for playground experimentation
    
    Alternative configurations: "cuda" for NVIDIA GPUs, "cpu" for CPU-only
    """


def main():
    """
    Execute minimal DiLoCo distributed training experiment.
    
    This function provides the simplest possible demonstration of DiLoCo
    distributed training on a GPT model. Designed for educational purposes
    and rapid experimentation with distributed training concepts.
    
    Experiment Overview:
    1. Load OpenWebText dataset with conservative data allocation
    2. Create moderate-sized GPT model (8 layers, 512 embedding)
    3. Configure DiLoCo strategy with H=200 communication interval
    4. Execute distributed training across 4 nodes with MPS acceleration
    5. Monitor training progress with frequent validation
    
    Key Learning Objectives:
    - Understanding DiLoCo communication patterns and benefits
    - Observing distributed training coordination and synchronization
    - Comparing DiLoCo behavior with standard distributed training
    - Experimenting with distributed training hyperparameters
    
    Modification Points:
    - PlaygroundConfig.NUM_NODES: Change distributed node count
    - PlaygroundConfig.DILOCO_COMMUNICATION_INTERVAL: Adjust H parameter
    - PlaygroundConfig.LEARNING_RATE: Modify optimization speed
    - PlaygroundConfig.MAX_STEPS: Control experiment duration
    """
    # Dataset Loading and Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    # Load OpenWebText dataset with distributed node configuration
    # First-time execution will download and process the dataset automatically
    train_dataset, vocab_size = get_dataset(
        PlaygroundConfig.DATASET_NAME,
        block_size=PlaygroundConfig.BLOCK_SIZE,
        device="cpu",
        start_pc=0.0,
        end_pc=PlaygroundConfig.DATA_PERCENTAGE_PER_NODE * PlaygroundConfig.NUM_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        PlaygroundConfig.DATASET_NAME,
        block_size=PlaygroundConfig.BLOCK_SIZE,
        device="cpu",
        start_pc=PlaygroundConfig.VALIDATION_START_PERCENTAGE,
        end_pc=PlaygroundConfig.VALIDATION_END_PERCENTAGE
    )

    # Model Architecture Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Create GPT model with playground-optimized architecture
    # Balanced configuration for educational demonstration and reasonable training speed
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=PlaygroundConfig.BLOCK_SIZE,
        n_layer=PlaygroundConfig.GPT_LAYERS,
        n_head=PlaygroundConfig.GPT_HEADS,
        n_embd=PlaygroundConfig.GPT_EMBEDDING_DIM,
        dropout=PlaygroundConfig.DROPOUT_RATE,
    )
    model = GPT(gpt_config)

    # Distributed Training Coordinator Setup
    # ─────────────────────────────────────────────────────────────────────────
    # LocalTrainer handles multi-node coordination and communication
    # Default port assignment avoids conflicts with other distributed training jobs
    trainer = LocalTrainer(
        model,
        train_dataset,
        val_dataset,
        # port=12355  # Uncomment and modify if port conflicts occur
    )

    # DiLoCo Strategy Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Configure DiLoCo distributed training strategy with educational parameters
    # H=200 provides good balance between communication efficiency and convergence
    from exogym.strategy.diloco import DiLoCoStrategy

    strategy = DiLoCoStrategy(
        optim_spec=OptimSpec(
            torch.optim.AdamW, 
            lr=PlaygroundConfig.LEARNING_RATE
        ),
        lr_scheduler="lambda_cosine",
        lr_scheduler_kwargs={
            "warmup_steps": PlaygroundConfig.WARMUP_STEPS,
            "cosine_anneal": PlaygroundConfig.COSINE_ANNEALING,
        },
        max_norm=PlaygroundConfig.GRADIENT_CLIP_NORM,
        H=PlaygroundConfig.DILOCO_COMMUNICATION_INTERVAL,
    )

    # Distributed Training Execution
    # ─────────────────────────────────────────────────────────────────────────
    # Execute DiLoCo distributed training with educational monitoring
    # Frequent validation provides detailed progress tracking for learning
    trainer.fit(
        num_epochs=PlaygroundConfig.NUM_EPOCHS,
        max_steps=PlaygroundConfig.MAX_STEPS,
        strategy=strategy,
        num_nodes=PlaygroundConfig.NUM_NODES,
        device=PlaygroundConfig.DEVICE,
        batch_size=PlaygroundConfig.BATCH_SIZE,
        minibatch_size=PlaygroundConfig.MINIBATCH_SIZE,
        shuffle=False,  # Maintain dataset ordering for reproducible experiments
        val_size=PlaygroundConfig.VALIDATION_SIZE,
        val_interval=PlaygroundConfig.VALIDATION_INTERVAL,
        # wandb_project='DiLoCo-Playground',  # Uncomment for experiment tracking
        # run_name='diloco-playground-demo'   # Uncomment for run identification
    )


if __name__ == "__main__":
    main()
