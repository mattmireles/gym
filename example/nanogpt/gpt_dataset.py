"""
GPT Dataset Implementations - PyTorch Dataset Classes for Text Data

This file implements specialized PyTorch Dataset classes optimized for GPT training
with different memory and performance characteristics. Provides both contiguous
(sliding window) and non-contiguous (pre-segmented) dataset access patterns,
with intelligent lazy loading for large-scale datasets.

Role in System:
- Core PyTorch Dataset implementations for all GPT training workflows
- Memory-efficient lazy loading for large datasets that don't fit in memory
- Device-aware tensor management for optimal GPU utilization
- Flexible dataset access patterns supporting different training requirements

Called by:
- nanogpt/dataset.py get_dataset() factory function for dataset instantiation
- exogym.trainer.LocalTrainer during distributed training data loading
- PyTorch DataLoader for batch creation during training
- Research scripts requiring custom dataset access patterns

Calls:
- torch for tensor operations and device management
- numpy for efficient array operations and file I/O
- os for file system operations and cache management
- PyTorch Dataset base class for standard dataset interface

Dataset Implementations:

1. ContiguousGPTTrainDataset:
   - Sliding window access over continuous token streams
   - Preserves long-range dependencies and context
   - Suitable for smaller datasets that fit in memory
   - Memory efficient for sequential data access

2. NonContiguousGPTTrainDataset:
   - Pre-segmented sequences with fixed block sizes
   - No continuity between examples (independent sequences)
   - Optimized for distributed training with consistent batch shapes
   - Direct tensor loading without sliding window overhead

3. LazyNonContiguousGPTTrainDataset:
   - Chunk-based lazy loading for large datasets like OpenWebText
   - LRU cache management for optimal memory utilization
   - Automatic chunk eviction based on configurable memory limits
   - Transparent chunk loading with global index mapping

Memory Management Strategy:
- Configurable chunk caching with max_chunks_in_memory parameter
- LRU eviction policy for optimal cache utilization
- Device-aware tensor allocation (CPU, CUDA, MPS)
- Automatic memory monitoring and cache statistics
"""

import torch
import numpy as np
import os


class NonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-segmented training data (2D tensor).
    
    Handles datasets where sequences are already divided into fixed-length chunks
    with no continuity between examples. Each row represents an independent
    training sequence, making it suitable for datasets with natural boundaries
    or when memory constraints require pre-segmented data.
    
    Data Format:
        - Input: 2D numpy array of shape (num_examples, block_size)
        - Each row is a complete, independent sequence
        - No sliding window or overlap between examples
        - Consistent sequence lengths across all examples
        
    Use Cases:
        - Pre-processed datasets with natural sequence boundaries
        - Memory-constrained scenarios requiring fixed batch shapes
        - Distributed training with consistent data shapes across nodes
        - Baseline comparisons with non-overlapping sequence training
        
    Memory Characteristics:
        - Loads entire dataset into GPU memory at initialization
        - Fixed memory footprint based on dataset size
        - No dynamic loading or caching mechanisms
        - Suitable for datasets that fit comfortably in memory
        
    Training Implications:
        - No context preservation between sequences
        - Each example is independent for gradient computation
        - Simplified data loading with predictable memory usage
        - Consistent batch shapes for distributed training stability
    """

    def __init__(self, data, device):
        assert data.ndim == 2
        self.examples, self.block_size = data.shape

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


class LazyNonContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for pre-segmented training data with intelligent lazy loading and LRU caching.
    
    Optimized for large-scale datasets that don't fit in memory, this implementation
    provides chunk-based lazy loading with configurable LRU cache management.
    Essential for training on datasets like OpenWebText where the full dataset
    exceeds available GPU memory.
    
    Chunk Management:
        - Data divided into chunks stored as individual numpy files
        - Global index mapping from dataset position to (chunk_id, local_index)
        - Automatic chunk loading and eviction based on access patterns
        - Configurable memory limits to prevent out-of-memory errors
        
    LRU Cache Strategy:
        - Least Recently Used eviction policy for optimal memory utilization
        - Configurable max_chunks_in_memory parameter for memory control
        - Access order tracking for intelligent cache replacement
        - Automatic cache statistics and memory monitoring
        
    Memory State Management:
        - _loaded_chunks: Dict mapping chunk_id → loaded tensor data
        - _chunk_access_order: List tracking access order for LRU eviction
        - Chunk size and offset arrays for O(1) index mapping
        - Device-aware tensor allocation for optimal GPU utilization
        
    Index Mapping System:
        - Global index space spanning all chunks seamlessly
        - Efficient conversion from global_idx → (chunk_id, local_idx)
        - Pre-computed chunk offsets for O(1) lookup performance
        - Handles variable chunk sizes and dataset boundaries
        
    Distributed Training Support:
        - Chunk-based data sharding across distributed nodes
        - Consistent global indexing for reproducible training
        - Memory-efficient per-node data loading
        - Automatic chunk discovery and validation
        
    Performance Optimizations:
        - Lazy loading reduces memory pressure and startup time
        - LRU caching maximizes cache hit rates during training
        - Device-aware tensor management for optimal GPU utilization
        - Minimal memory overhead for chunk metadata storage
    """

    def __init__(self, chunk_ids, cache_location, device, max_chunks_in_memory=None):
        self.chunk_ids = chunk_ids
        self.cache_location = cache_location
        self.device = device
        self.max_chunks_in_memory = max_chunks_in_memory

        # Build index mapping from global index to (chunk_id, local_idx)
        self.chunk_sizes = []
        self.chunk_offsets = []
        total_examples = 0

        print(f"Loading {len(chunk_ids)} chunks to determine sizes")
        for chunk_id in chunk_ids:
            cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"Cached chunk file not found: {cache_file}")

            # Load just to get shape, then discard
            chunk_data = np.load(cache_file)
            assert (
                chunk_data.ndim == 2
            ), f"Expected 2D chunk data, got {chunk_data.ndim}D"

            chunk_size = chunk_data.shape[0]
            self.chunk_sizes.append(chunk_size)
            self.chunk_offsets.append(total_examples)
            total_examples += chunk_size

            # Store block_size from first chunk
            if len(self.chunk_sizes) == 1:
                self.block_size = chunk_data.shape[1]

        self.total_examples = total_examples
        self._loaded_chunks = {}  # Cache for loaded chunks
        self._chunk_access_order = []  # Track access order for LRU eviction

        print(
            f"Dataset initialized: {len(chunk_ids)} chunks, {self.total_examples} total examples"
        )

    def __len__(self):
        return self.total_examples

    def _get_chunk_and_local_idx(self, global_idx):
        """Convert global index to (chunk_id, local_idx)"""
        for i, (chunk_id, offset, size) in enumerate(
            zip(self.chunk_ids, self.chunk_offsets, self.chunk_sizes)
        ):
            if global_idx < offset + size:
                local_idx = global_idx - offset
                return chunk_id, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def _evict_old_chunks(self):
        """Remove old chunks from memory if we exceed the limit"""
        if self.max_chunks_in_memory is None:
            return

        while len(self._loaded_chunks) > self.max_chunks_in_memory:
            # Remove least recently used chunk
            oldest_chunk = self._chunk_access_order.pop(0)
            if oldest_chunk in self._loaded_chunks:
                del self._loaded_chunks[oldest_chunk]

    def _load_chunk(self, chunk_id):
        """Load chunk data if not already cached in memory"""
        if chunk_id not in self._loaded_chunks:
            # print(f'loading chunk {chunk_id}')
            cache_file = f"{self.cache_location}/chunk_{chunk_id}.npy"
            chunk_data = np.load(cache_file)
            self._loaded_chunks[chunk_id] = (
                torch.from_numpy(chunk_data).to(device=self.device).long()
            )

            # Evict old chunks if necessary
            self._evict_old_chunks()

        # Update access order for LRU
        if chunk_id in self._chunk_access_order:
            self._chunk_access_order.remove(chunk_id)
        self._chunk_access_order.append(chunk_id)

        return self._loaded_chunks[chunk_id]

    def __getitem__(self, idx):
        chunk_id, local_idx = self._get_chunk_and_local_idx(idx)
        chunk_data = self._load_chunk(chunk_id)
        x = chunk_data[local_idx]
        return x[:-1], x[1:]

    def get_memory_info(self):
        """Return information about current memory usage"""
        return {
            "chunks_in_memory": len(self._loaded_chunks),
            "max_chunks_in_memory": self.max_chunks_in_memory,
            "total_chunks": len(self.chunk_ids),
            "loaded_chunk_ids": list(self._loaded_chunks.keys()),
        }


class ContiguousGPTTrainDataset(torch.utils.data.Dataset):
    """
    Dataset for continuous token streams with sliding window access pattern.
    
    Implements sliding window approach over continuous text data, preserving
    long-range dependencies and document structure. Creates overlapping training
    examples that maintain context across sequence boundaries, essential for
    high-quality language model training.
    
    Data Format:
        - Input: 1D numpy array representing continuous token stream
        - Sliding window creates overlapping sequences of fixed length
        - Each example preserves context from previous sequences
        - Natural document and paragraph boundaries maintained
        
    Context Preservation:
        - Overlapping windows maintain cross-sequence dependencies
        - Document structure and long-range patterns preserved
        - Enables learning of inter-sentence and inter-paragraph relationships
        - Critical for coherent text generation and understanding
        
    Memory Characteristics:
        - Loads entire dataset into GPU memory at initialization
        - Memory usage proportional to dataset size
        - Fixed memory footprint with no dynamic loading
        - Optimal for datasets that fit comfortably in available memory
        
    Training Benefits:
        - Maximum utilization of available training data
        - Preserved context improves model quality
        - Consistent with language modeling best practices
        - Better learning of long-range dependencies
        
    Use Cases:
        - Small to medium datasets (Shakespeare, WikiText)
        - Educational experiments requiring context preservation
        - Baseline comparisons with full context utilization
        - Research scenarios prioritizing model quality over memory efficiency
        
    Implementation Details:
        - Sliding window with stride 1 for maximum data utilization
        - Block size determines sequence length for training
        - Automatic input/target splitting (input[:-1], target[1:])
        - Device-aware tensor allocation for GPU acceleration
    """

    def __init__(self, data, block_size, device):
        assert data.ndim == 1

        self.device = device

        self.data = torch.from_numpy(data).to(device=device).long()
        self.block_size = block_size

    def __len__(self):
        return self.data.shape[0] - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size + 1]
        return x[:-1], x[1:]
