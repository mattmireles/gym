"""
NanoGPT Module - GPT Implementation and Dataset Utilities

This module provides a complete GPT implementation optimized for distributed training
research and experimentation. Includes model architecture, dataset processing,
and utilities specifically designed for exogym distributed training framework.

Module Components:
- GPT model implementation with configurable architecture sizes
- Dataset processing pipeline supporting multiple text datasets  
- Lazy loading mechanisms for large-scale datasets like OpenWebText
- Efficient tokenization and preprocessing for distributed training

Role in System:
- Core GPT model implementation used by all distributed training examples
- Dataset factory providing consistent interfaces for text dataset loading
- Preprocessing utilities optimized for distributed training workflows
- Configuration management for model sizes and training hyperparameters

Called by:
- example/nanogpt.py for comprehensive distributed training experiments
- example/playground.py for simple DiLoCo training demonstrations
- example/diloco_scaling_batchsize.py for batch size scaling experiments
- External scripts importing GPT model and dataset utilities

Exports:
- GPT: Main transformer model implementation with distributed training support
- GPTConfig: Configuration dataclass with predefined model size configurations
- get_dataset: Universal dataset loading function supporting multiple formats
- build_dataset_small: Processing for smaller datasets (shakespeare, wikitext)
- build_dataset_owt: Optimized processing for OpenWebText large-scale dataset

Design Philosophy:
- Prioritizes simplicity and clarity for research experimentation
- Optimized for distributed training with minimal memory overhead
- Supports both small-scale educational examples and large-scale research
- Provides flexible configuration while maintaining sensible defaults
"""

# nanogpt module - GPT implementation and dataset utilities for distributed training

from .build_dataset import build_dataset_small, build_dataset_owt
from .dataset import get_dataset
from .nanogpt import GPT, GPTConfig

__all__ = ["get_dataset", "build_dataset_small", "build_dataset_owt", "GPT", "GPTConfig"]
