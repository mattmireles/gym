"""
MNIST Distributed Training Strategy Comparison

This file implements a comprehensive comparison of distributed training strategies
(DiLoCo, SPARTA, SimpleReduce) on the MNIST dataset using a robust CNN architecture.
Designed to validate strategy effectiveness on a well-understood computer vision task
before scaling to larger language models.

Role in System:
- Provides a controlled environment for testing distributed training strategies
- Uses MNIST as a standardized benchmark for strategy comparison
- Implements a production-grade CNN with proper regularization and data augmentation
- Serves as a template for strategy comparison on other vision tasks

Called by:
- Researchers comparing distributed training strategies
- Integration tests for new distributed training implementations
- Educational demonstrations of distributed training concepts
- Automated benchmarking systems

Calls:
- exogym.trainer.LocalTrainer for distributed training coordination
- exogym.strategy.diloco.DiLoCoStrategy for distributed learning coordination
- exogym.strategy.sparta.SPARTAStrategy for sparse communication training
- exogym.strategy.strategy.SimpleReduceStrategy for standard parameter averaging
- PyTorch torchvision for MNIST dataset loading and preprocessing

Architecture Details:
- Two-block CNN with BatchNorm, ReLU, MaxPool, and Dropout layers
- First block: 1→64→64 channels with 3x3 convolutions
- Second block: 64→128→128 channels with 3x3 convolutions
- Classifier: 128*7*7 → 256 → 10 with dropout regularization
- Data augmentation: Random affine transformations and normalization

Training Configuration:
- Tests all three strategies with identical hyperparameters for fair comparison
- Uses 4 nodes, 5 epochs, batch size 256 for comprehensive evaluation
- Includes proper validation on full 10k test set for statistical significance
- H=10 communication interval optimized for MNIST's fast convergence
"""

# mnist_compare_strategies_big.py  (2-space indent preserved ✨)
from exogym.trainer import LocalTrainer
from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.sparta import SPARTAStrategy
from exogym.strategy.strategy import SimpleReduceStrategy
from exogym.strategy.optim import OptimSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split


class MNISTExperimentConfig:
    """
    MNIST distributed training experiment configuration constants.
    
    This class centralizes all experimental parameters for MNIST distributed
    training strategy comparison. Values are optimized for MNIST's characteristics:
    fast convergence, small dataset size, and well-understood optimization landscape.
    """
    
    # ── Dataset Configuration ──────────────────────────────────────────────
    
    MNIST_INPUT_CHANNELS = 1
    """Number of input channels for MNIST images.
    
    MNIST consists of grayscale images, hence single channel input.
    This constant ensures architecture consistency and clear documentation.
    """
    
    MNIST_NUM_CLASSES = 10
    """Number of classification classes in MNIST dataset.
    
    MNIST contains digits 0-9, requiring 10-class classification.
    Used for final layer output dimension and loss computation.
    """
    
    MNIST_IMAGE_SIZE = 28
    """MNIST image dimension (28x28 pixels).
    
    Standard MNIST image size used for:
    - Data augmentation parameter calculations
    - Architecture dimension planning
    - Memory estimation and batch size optimization
    """
    
    # ── Data Augmentation Configuration ──────────────────────────────────────
    
    AUGMENTATION_ROTATION_DEGREES = 10
    """Maximum rotation angle for random affine augmentation.
    
    Conservative 10-degree rotation chosen to:
    - Provide meaningful augmentation without losing digit recognizability
    - Maintain MNIST digit orientation characteristics
    - Balance between regularization and data corruption
    """
    
    AUGMENTATION_TRANSLATION_FRACTION = 0.1
    """Translation fraction for random affine augmentation.
    
    10% translation in both x and y directions:
    - Simulates natural writing variation and camera positioning
    - Provides effective regularization without excessive distortion
    - Maintains digit centering while adding positional robustness
    """
    
    MNIST_NORMALIZATION_MEAN = 0.1307
    """MNIST dataset normalization mean value.
    
    Calculated from full MNIST training set statistics.
    Standard value used across MNIST research for consistent preprocessing.
    """
    
    MNIST_NORMALIZATION_STD = 0.3081
    """MNIST dataset normalization standard deviation.
    
    Calculated from full MNIST training set statistics.
    Ensures proper input scaling for neural network optimization.
    """
    
    # ── CNN Architecture Configuration ────────────────────────────────────────
    
    CNN_BLOCK1_CHANNELS = 64
    """Number of channels in first CNN block.
    
    Chosen as 64 to provide:
    - Sufficient feature extraction capacity for edge and corner detection
    - Reasonable parameter count for efficient distributed training
    - Good balance between expressiveness and computational cost
    - Standard choice following successful CNN architectures
    """
    
    CNN_BLOCK2_CHANNELS = 128
    """Number of channels in second CNN block.
    
    Doubled from block 1 (64→128) following standard practice:
    - Compensates for spatial dimension reduction from pooling
    - Increases representational capacity for higher-level features
    - Maintains computational balance as spatial dimensions decrease
    - Enables hierarchical feature learning progression
    """
    
    CNN_CLASSIFIER_HIDDEN_DIM = 256
    """Hidden dimension for CNN classifier head.
    
    Chosen as 256 to:
    - Provide sufficient capacity for combining CNN features
    - Avoid overfitting on MNIST's relatively simple patterns
    - Balance model expressiveness with training efficiency
    - Enable effective distributed training across multiple nodes
    """
    
    # ── Regularization Configuration ──────────────────────────────────────────
    
    CNN_CONV_DROPOUT = 0.25
    """Dropout rate for convolutional layers.
    
    Conservative 25% dropout chosen for:
    - Effective regularization without hampering feature learning
    - Maintaining spatial structure in convolutional feature maps
    - Preventing overfitting while preserving learning capacity
    - Stability across different distributed training configurations
    """
    
    CNN_CLASSIFIER_DROPOUT = 0.5
    """Dropout rate for fully connected classifier layers.
    
    Higher 50% dropout for dense layers:
    - Stronger regularization needed for fully connected layers
    - Prevents memorization of training patterns
    - Follows established best practices for CNN classification heads
    - Maintains generalization across distributed training scenarios
    """
    
    # ── Training Configuration ──────────────────────────────────────────────
    
    TRAINING_EPOCHS = 5
    """Number of training epochs for MNIST experiments.
    
    Conservative 5 epochs chosen because:
    - MNIST converges quickly, avoiding unnecessary computation
    - Sufficient for meaningful strategy comparison
    - Prevents overfitting while demonstrating convergence patterns
    - Reasonable experiment duration for distributed training validation
    """
    
    BATCH_SIZE = 256
    """Global batch size for distributed training.
    
    Chosen as 256 to:
    - Provide stable gradient estimates for optimization
    - Enable efficient distributed training across multiple nodes
    - Balance memory usage with convergence speed
    - Support clean division across different node counts
    """
    
    MINIBATCH_SIZE = 256
    """Minibatch size for gradient accumulation.
    
    Set equal to batch size (no gradient accumulation) for:
    - Simplicity in experimental comparison
    - Sufficient memory availability for MNIST's small images
    - Clean experimental conditions without accumulation complexity
    - Direct batch size control across distributed configurations
    """
    
    # ── Distributed Training Configuration ────────────────────────────────────
    
    NUM_NODES = 4
    """Number of distributed training nodes.
    
    Chosen as 4 nodes to:
    - Demonstrate meaningful distributed training scaling
    - Test strategy effectiveness across multiple workers
    - Balance experimental complexity with resource requirements
    - Enable clean batch size division (256 / 4 = 64 per node)
    """
    
    COMMUNICATION_INTERVAL = 10
    """Communication interval (H) for distributed strategies.
    
    Short interval of 10 steps chosen for MNIST because:
    - MNIST converges quickly, requiring frequent synchronization
    - Small dataset benefits from more frequent parameter averaging
    - Prevents divergence in fast-convergence scenarios
    - Optimized for MNIST's unique training characteristics vs large LMs
    """
    
    # ── Optimization Configuration ──────────────────────────────────────────
    
    LEARNING_RATE = 3e-4
    """Learning rate for AdamW optimizer.
    
    Conservative 3e-4 chosen for:
    - Stable training across different distributed configurations
    - Good convergence properties for CNN architectures
    - Robustness to communication delays in distributed training
    - Balance between training speed and stability
    """
    
    WEIGHT_DECAY = 1e-4
    """Weight decay for L2 regularization.
    
    Light regularization (1e-4) chosen to:
    - Prevent overfitting without hindering learning
    - Maintain model capacity for feature extraction
    - Provide consistent regularization across distributed nodes
    - Follow established practices for CNN training
    """
    
    WARMUP_STEPS = 100
    """Number of learning rate warmup steps.
    
    Short warmup for MNIST because:
    - Fast convergence doesn't require extensive warmup
    - Prevents initial training instability
    - Maintains consistent optimization across distributed strategies
    - Appropriate scale for MNIST's training characteristics
    """
    
    # ── Validation Configuration ────────────────────────────────────────────
    
    VALIDATION_INTERVAL = 10
    """Validation evaluation interval in training steps.
    
    Frequent validation every 10 steps for:
    - Detailed monitoring of fast MNIST convergence
    - Strategy comparison with high temporal resolution
    - Early detection of training issues or divergence
    - Comprehensive loss curve analysis for research
    """


# ── 1. Dataset ───────────────────────────────────────────────────────────────
def get_mnist_splits(root="data", train_frac=1.0):
    """
    Load MNIST training dataset with augmentation and proper normalization.
    
    Creates training dataset with data augmentation transforms optimized for
    distributed training robustness. Uses standard MNIST normalization values
    and conservative augmentation to maintain digit recognizability.
    
    Args:
        root: Directory to store/load MNIST dataset files
        train_frac: Fraction of training data to use (1.0 = full dataset)
    
    Returns:
        Tuple of (training_dataset, validation_dataset) or (full_dataset, None)
    
    Transform Pipeline:
        1. RandomAffine: Rotation + translation for regularization
        2. ToTensor: Convert PIL Image to torch.Tensor
        3. Normalize: Zero-mean, unit-variance using MNIST statistics
    """
    # Data augmentation and preprocessing pipeline
    # Optimized for distributed training stability and generalization
    augmentation_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=MNISTExperimentConfig.AUGMENTATION_ROTATION_DEGREES,
            translate=(
                MNISTExperimentConfig.AUGMENTATION_TRANSLATION_FRACTION,
                MNISTExperimentConfig.AUGMENTATION_TRANSLATION_FRACTION
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (MNISTExperimentConfig.MNIST_NORMALIZATION_MEAN,),
            (MNISTExperimentConfig.MNIST_NORMALIZATION_STD,)
        ),
    ])
    
    # Load MNIST training set with augmentation transforms
    full_dataset = datasets.MNIST(
        root, 
        train=True, 
        download=True, 
        transform=augmentation_transform
    )
    
    # Optional dataset fraction for ablation studies or quick testing
    if train_frac < 1.0:
        n_train = int(len(full_dataset) * train_frac)
        n_val = len(full_dataset) - n_train
        return random_split(full_dataset, [n_train, n_val])
    
    return full_dataset, None  # Validation set handled separately in run_sweep()


# ── 2. Stronger CNN ───────────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    Production-grade CNN architecture for MNIST classification.
    
    Implements a two-block CNN with BatchNorm, ReLU activation, MaxPooling,
    and Dropout regularization. Architecture designed for:
    - Robust feature extraction with hierarchical learning
    - Effective regularization to prevent overfitting
    - Efficient distributed training across multiple nodes
    - Balance between model capacity and computational efficiency
    
    Architecture Flow:
        Input (1×28×28) → Block1 (64 channels, 14×14) → Block2 (128 channels, 7×7) → Classifier → Output (10 classes)
    
    Each block follows the pattern:
        Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool → Dropout
    """
    
    def __init__(self):
        super().__init__()
        
        # Feature extraction layers with hierarchical channel progression
        self.features = nn.Sequential(
            # ── Convolutional Block 1: 28×28 → 14×14 ───────────────────────
            # First conv layer: Extract basic features (edges, corners)
            nn.Conv2d(
                MNISTExperimentConfig.MNIST_INPUT_CHANNELS, 
                MNISTExperimentConfig.CNN_BLOCK1_CHANNELS, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(MNISTExperimentConfig.CNN_BLOCK1_CHANNELS),
            nn.ReLU(inplace=True),
            
            # Second conv layer: Refine features within block
            nn.Conv2d(
                MNISTExperimentConfig.CNN_BLOCK1_CHANNELS, 
                MNISTExperimentConfig.CNN_BLOCK1_CHANNELS, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(MNISTExperimentConfig.CNN_BLOCK1_CHANNELS),
            nn.ReLU(inplace=True),
            
            # Spatial reduction and regularization
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28×28 → 14×14
            nn.Dropout2d(MNISTExperimentConfig.CNN_CONV_DROPOUT),
            
            # ── Convolutional Block 2: 14×14 → 7×7 ─────────────────────────
            # Third conv layer: Higher-level feature extraction
            nn.Conv2d(
                MNISTExperimentConfig.CNN_BLOCK1_CHANNELS, 
                MNISTExperimentConfig.CNN_BLOCK2_CHANNELS, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(MNISTExperimentConfig.CNN_BLOCK2_CHANNELS),
            nn.ReLU(inplace=True),
            
            # Fourth conv layer: Complex pattern recognition
            nn.Conv2d(
                MNISTExperimentConfig.CNN_BLOCK2_CHANNELS, 
                MNISTExperimentConfig.CNN_BLOCK2_CHANNELS, 
                kernel_size=3, 
                padding=1
            ),
            nn.BatchNorm2d(MNISTExperimentConfig.CNN_BLOCK2_CHANNELS),
            nn.ReLU(inplace=True),
            
            # Final spatial reduction and regularization
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14×14 → 7×7
            nn.Dropout2d(MNISTExperimentConfig.CNN_CONV_DROPOUT),
        )
        
        # Classification head with feature combination and output projection
        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            # Feature combination layer
            nn.Linear(
                MNISTExperimentConfig.CNN_BLOCK2_CHANNELS * 7 * 7,  # 128 * 7 * 7 = 6272
                MNISTExperimentConfig.CNN_CLASSIFIER_HIDDEN_DIM
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(MNISTExperimentConfig.CNN_CLASSIFIER_DROPOUT),
            
            # Output projection to class logits
            nn.Linear(
                MNISTExperimentConfig.CNN_CLASSIFIER_HIDDEN_DIM,
                MNISTExperimentConfig.MNIST_NUM_CLASSES
            ),
        )

    def forward(self, x):
        """
        Forward pass through CNN architecture.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Logits tensor of shape (batch_size, 10) for MNIST classification
        """
        features = self.features(x)
        return self.classifier(features)


# ── 3. Wrapper (returns logits, loss) ─────────────────────────────────────────
class ModelWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        imgs, labels = batch
        logits = self.backbone(imgs)
        return F.cross_entropy(logits, labels)


# ── 4. Training sweep ─────────────────────────────────────────────────────────
def run_sweep():
    """
    Execute comprehensive comparison of distributed training strategies on MNIST.
    
    This function implements a controlled experiment comparing three distributed
    training strategies (DiLoCo, SPARTA, SimpleReduce) on MNIST classification.
    Uses identical model architecture, optimization settings, and training
    configuration to ensure fair comparison.
    
    Experimental Design:
    - Same CNN architecture for all strategies
    - Identical optimization hyperparameters  
    - Full MNIST test set validation for statistical significance
    - Conservative training duration to prevent overfitting
    - Comprehensive logging for strategy performance analysis
    
    Strategy Comparison:
    - DiLoCo: Distributed Learning Coordination with outer loop optimization
    - SPARTA: Sparse communication training with gradient sparsification
    - SimpleReduce: Standard distributed data parallel (DDP) baseline
    """
    # Dataset Preparation with Consistent Preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    # Load training dataset with augmentation for robustness
    train_ds, _ = get_mnist_splits()
    
    # Load validation dataset with normalization only (no augmentation)
    # Uses test set for unbiased final evaluation
    val_ds = datasets.MNIST(
        "data",
        train=False,  # Use test set for validation
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (MNISTExperimentConfig.MNIST_NORMALIZATION_MEAN,),
                (MNISTExperimentConfig.MNIST_NORMALIZATION_STD,)
            )
        ]),
    )
    
    # Device Selection with Apple Silicon MPS Support
    # ─────────────────────────────────────────────────────────────────────────
    # Automatic device selection prioritizing GPU acceleration
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() 
        else "cpu"
    )
    
    # Optimization Configuration Shared Across All Strategies
    # ─────────────────────────────────────────────────────────────────────────
    # Conservative hyperparameters for stable distributed training comparison
    optim_spec = OptimSpec(
        torch.optim.AdamW, 
        lr=MNISTExperimentConfig.LEARNING_RATE,
        weight_decay=MNISTExperimentConfig.WEIGHT_DECAY
    )

    # Strategy Comparison Loop
    # ─────────────────────────────────────────────────────────────────────────
    # Test each distributed training strategy with identical configuration
    strategy_configurations = [
        ("diloco", DiLoCoStrategy),
        ("sparta", SPARTAStrategy),
        ("simplereduce", SimpleReduceStrategy),
    ]
    
    for strategy_name, StrategyClass in strategy_configurations:
        # Model Instantiation with Fresh Initialization
        # ─────────────────────────────────────────────────────────────────────
        # Create new model instance for each strategy to ensure fair comparison
        model = ModelWrapper(CNN())
        trainer = LocalTrainer(model, train_ds, val_ds)

        # Strategy-Specific Configuration
        # ─────────────────────────────────────────────────────────────────────
        # All strategies use identical base configuration with strategy-specific parameters
        strategy = StrategyClass(
            optim_spec=optim_spec,
            H=MNISTExperimentConfig.COMMUNICATION_INTERVAL,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={
                "warmup_steps": MNISTExperimentConfig.WARMUP_STEPS,
                "cosine_anneal": True
            },
        )

        # Distributed Training Execution
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n=== {strategy_name.upper()} DISTRIBUTED TRAINING ===")
        trainer.fit(
            num_epochs=MNISTExperimentConfig.TRAINING_EPOCHS,
            strategy=strategy,
            num_nodes=MNISTExperimentConfig.NUM_NODES,
            device=device,
            batch_size=MNISTExperimentConfig.BATCH_SIZE,
            minibatch_size=MNISTExperimentConfig.MINIBATCH_SIZE,
            val_size=len(val_ds),  # Full 10,000 test set for robust evaluation
            val_interval=MNISTExperimentConfig.VALIDATION_INTERVAL,
            # wandb_project="mnist-strategy-comparison",  # Enable for experiment tracking
            run_name=f"{strategy_name}_mnist_distributed",
        )


if __name__ == "__main__":
    run_sweep()
