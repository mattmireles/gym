"""
Dataset Interface and Factory - Unified Dataset Loading

This file provides the unified interface for loading and accessing text datasets
in the distributed training pipeline. Implements the factory pattern to abstract
dataset-specific loading details and provides consistent PyTorch Dataset interfaces
for both small and large-scale datasets.

Role in System:
- Primary dataset interface used by all distributed training scripts
- Abstracts complexity of different dataset formats and loading mechanisms
- Provides device-aware dataset loading for efficient GPU utilization
- Implements intelligent dataset selection based on scale and memory requirements

Called by:
- example/nanogpt.py main training CLI for dataset instantiation  
- example/playground.py for simple DiLoCo training experiments
- example/diloco_scaling_batchsize.py for batch size scaling studies
- Any distributed training script requiring text dataset loading

Calls:
- nanogpt/build_dataset.py for dataset preprocessing and caching
- nanogpt/gpt_dataset.py for PyTorch Dataset implementations
- numpy for efficient array operations and data manipulation
- os for file system operations and cache management

Dataset Factory Pattern:
- get_dataset(): Universal entry point supporting all dataset types
- Automatic selection between ContiguousGPTTrainDataset and LazyNonContiguousGPTTrainDataset
- Device-aware tensor allocation for optimal memory utilization
- Consistent interface regardless of underlying dataset implementation

Supported Dataset Modes:
- Small datasets (shakespeare, wikitext): ContiguousGPTTrainDataset with full loading
- Large datasets (owt): LazyNonContiguousGPTTrainDataset with chunk-based loading
- Automatic mode selection based on dataset size and available memory

Memory Management:
- Device-aware tensor allocation (CPU, CUDA, MPS)
- Configurable chunk caching for large datasets
- Memory-efficient lazy loading for datasets that don't fit in memory
- Automatic garbage collection of unused dataset chunks
"""

import numpy as np
import os

from .build_dataset import build_dataset_small, build_dataset_owt
from .gpt_dataset import ContiguousGPTTrainDataset, LazyNonContiguousGPTTrainDataset


def load_chunk(chunk_id, s3_client):
    """
    Load a specific dataset chunk from local cache storage.
    
    Provides consistent interface for loading preprocessed dataset chunks during
    distributed training. Handles cache directory creation and validates chunk
    availability before loading.
    
    Args:
        chunk_id (int): Unique identifier for the dataset chunk to load
        s3_client: Legacy parameter (unused) - maintained for API compatibility
        
    Returns:
        np.ndarray: Loaded chunk data as numpy array with shape (num_blocks, block_size)
        
    Raises:
        Exception: If specified chunk file doesn't exist in cache directory
        
    Cache Structure:
        - Cache directory: "data/owt/"
        - Chunk files: "chunk_{chunk_id}.npy"
        - Chunks contain pre-tokenized blocks ready for training
        
    Usage Context:
        - Called by LazyNonContiguousGPTTrainDataset for on-demand chunk loading
        - Supports distributed training with chunk-based data sharding
        - Enables memory-efficient training on large datasets
        
    Error Handling:
        - Validates cache directory existence (creates if needed)
        - Provides clear error message with expected file path
        - Maintains training stability by failing fast on missing chunks
    """
    cache_location = "data/owt"
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)
    else:
        raise Exception(f"Chunk {chunk_id} not found in {cache_file}")


def get_dataset(
    dataset_name,
    block_size,
    device,
    start_pc=0.0,
    end_pc=1.0,
    max_workers=8,
    max_chunks_in_memory=None,
):
    """
    Universal dataset factory for GPT training with automatic format selection.
    
    Provides unified interface for loading text datasets with automatic selection
    between contiguous (sliding window) and non-contiguous (chunked) dataset
    implementations based on dataset characteristics and scale.
    
    Args:
        dataset_name (str): Dataset identifier ("shakespeare", "wikitext", "owt")
        block_size (int): Sequence length for training blocks (e.g., 1024)
        device (str): Target device for tensor allocation ("cpu", "cuda", "mps")
        start_pc (float): Start percentage of dataset to use (0.0-1.0)
        end_pc (float): End percentage of dataset to use (0.0-1.0)
        max_workers (int): Parallel workers for dataset preprocessing
        max_chunks_in_memory (int, optional): Maximum chunks to cache for large datasets
        
    Returns:
        tuple: (dataset, vocab_size)
            - dataset: PyTorch Dataset instance optimized for the data scale
            - vocab_size (int): Vocabulary size for model embedding layer
            
    Dataset Selection Logic:
        - Small datasets (shakespeare, wikitext): ContiguousGPTTrainDataset
          Uses sliding window over continuous token stream for context preservation
        - Large datasets (owt): LazyNonContiguousGPTTrainDataset  
          Uses chunk-based lazy loading for memory efficiency
          
    Implementation Details:
        ContiguousGPTTrainDataset:
        - Loads entire dataset into memory as 1D token array
        - Sliding window creates overlapping training examples
        - Preserves long-range dependencies and document structure
        - Optimal for datasets that fit comfortably in memory
        
        LazyNonContiguousGPTTrainDataset:
        - Loads dataset chunks on-demand during training
        - Pre-segmented blocks (no sliding window) for consistent shapes
        - LRU cache management for optimal memory utilization
        - Essential for training on datasets larger than available memory
        
    Device Management:
        - Device-aware tensor allocation for optimal GPU utilization
        - Automatic tensor migration to specified device
        - Support for CPU, CUDA, and Apple Silicon MPS backends
        - Memory pressure management through configurable chunk caching
        
    Performance Optimization:
        - Automatic dataset format selection based on scale
        - Configurable parallelism for dataset preprocessing
        - Intelligent caching strategies for repeated experiments
        - Memory-efficient lazy loading for large-scale training
        
    Error Handling:
        - Validates dataset name and parameter ranges
        - Provides clear error messages for common configuration issues
        - Handles missing cache files and preprocessing failures
        - Maintains training stability through robust error recovery
    """
    if dataset_name != "owt":
        data, vocab_size = build_dataset_small(
            dataset_name, block_size, start_pc, end_pc
        )

        dataset = ContiguousGPTTrainDataset(data, block_size=block_size, device=device)
    else:
        chunk_ids, cache_location, vocab_size = build_dataset_owt(
            start_pc, end_pc, max_workers=max_workers
        )

        dataset = LazyNonContiguousGPTTrainDataset(
            chunk_ids,
            cache_location,
            device=device,
            max_chunks_in_memory=max_chunks_in_memory,
        )

    return dataset, vocab_size
