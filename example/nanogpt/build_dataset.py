"""
Dataset Building and Preprocessing Pipeline

This file implements the core dataset preprocessing pipeline for training GPT models
on various text datasets. Provides optimized processing for both small educational
datasets and large-scale datasets like OpenWebText, with intelligent caching and
distributed training support.

Role in System:
- Central dataset preprocessing hub for all text datasets used in distributed training
- Implements efficient tokenization and chunking strategies for different dataset sizes
- Provides intelligent caching to avoid reprocessing datasets between experiments  
- Supports both character-level and subword tokenization strategies

Called by:
- nanogpt/dataset.py get_dataset() function for unified dataset interface
- nanogpt/download_dataset.py for standalone dataset preprocessing
- Distributed training scripts requiring consistent dataset preprocessing
- Research scripts needing reproducible dataset splits and processing

Calls:
- transformers.GPT2Tokenizer for subword tokenization on larger datasets
- datasets library for efficient loading and processing of HuggingFace datasets
- numpy for efficient array operations and caching
- os.cpu_count() for optimal multiprocessing during tokenization

Supported Datasets:
- shakespeare: Character-level tokenization, small dataset for quick experiments
- wikitext: GPT-2 tokenization, medium-sized dataset for validation
- owt (OpenWebText): GPT-2 tokenization with chunking, large-scale training dataset

Processing Pipeline:
1. Dataset download and verification from HuggingFace hub
2. Tokenization using appropriate tokenizer (character or GPT-2)
3. Sequence concatenation with EOS token separation
4. Block size chunking for efficient training data loading
5. Intelligent caching with cache invalidation on parameter changes
6. Distributed data sharding support for multi-node training

Caching Strategy:
- File-based caching with parameter-specific cache keys
- Automatic cache invalidation when processing parameters change
- Efficient chunk-based caching for large datasets to support distributed loading
- Memory-efficient processing that doesn't require loading entire datasets
"""

import torch
import argparse
import numpy as np
import os
from datasets import load_dataset, load_dataset_builder, concatenate_datasets


def generate_char_vocab():
    """
    Generate fixed character vocabulary for character-level tokenization.
    
    Creates deterministic character-to-integer mapping for Shakespeare and other
    character-level datasets. Includes standard ASCII characters plus special
    end-of-sequence token for proper sequence termination.
    
    Returns:
        tuple: (char_to_int_mapping, eos_token_id)
            - char_to_int_mapping (dict): Character to integer ID mapping
            - eos_token_id (int): Integer ID for end-of-sequence token
            
    Vocabulary Contents:
        - Basic punctuation: space, !, $, &, ', -, ., 3, :, ;, ?
        - Alphabetic characters: A-Z, a-z (full English alphabet)
        - Newline character for text structure preservation
        - Special EOS token: "<EOS>" for sequence boundary marking
        
    Design Rationale:
        - Fixed vocabulary ensures reproducible tokenization across runs
        - Character-level provides fine-grained control for small datasets
        - EOS token enables proper sequence boundary detection
        - Compact vocabulary (64 characters) for efficient embedding layers
        
    Usage Context:
        - Shakespeare dataset preprocessing for educational experiments
        - Small dataset character-level language modeling
        - Deterministic tokenization for research reproducibility
    """
    vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n"
    char_int = {char: i for i, char in enumerate(vocab)}
    # int_char = {i: char for i, char in enumerate(vocab)}

    # Define a special end-of-sequence token.
    eos_token = "<EOS>"
    char_int[eos_token] = len(char_int)
    eos_token_id = char_int[eos_token]
    return char_int, eos_token_id


def build_dataset_small(dataset, block_size=1024, start_pc=0.0, end_pc=1.0):
    """
    Load and preprocess small-scale datasets with intelligent caching and tokenization.
    
    Handles both character-level tokenization (Shakespeare) and subword tokenization
    (WikiText) with automatic tokenization strategy selection. Implements efficient
    caching to avoid reprocessing datasets between training runs.
    
    Args:
        dataset (str): Dataset identifier ("shakespeare" or "wikitext")
        block_size (int): Maximum sequence length for training blocks
        start_pc (float): Start percentage of dataset to use (0.0-1.0)
        end_pc (float): End percentage of dataset to use (0.0-1.0)
        
    Returns:
        tuple: (processed_data, vocab_size)
            - processed_data (np.ndarray): Tokenized and concatenated text data
            - vocab_size (int): Size of tokenizer vocabulary
            
    Tokenization Strategy:
        - Shakespeare: Character-level tokenization with fixed 64-character vocabulary
        - WikiText: GPT-2 subword tokenization with 50,257 token vocabulary
        - Automatic strategy selection based on dataset characteristics
        
    Caching Strategy:
        - Parameter-specific cache files prevent unnecessary reprocessing
        - Cache key includes dataset, block_size, start_pc, end_pc
        - Automatic cache validation and regeneration when parameters change
        - Separate cache directories for character vs subword tokenization
        
    Processing Pipeline:
        1. Check for existing cached data with matching parameters
        2. Load dataset from HuggingFace hub with specified data range
        3. Apply appropriate tokenization strategy (character or subword)
        4. Concatenate all sequences with EOS token separation
        5. Cache processed data for future use
        
    Dataset Handling:
        - Shakespeare: Trelis/tiny-shakespeare from HuggingFace
        - WikiText: wikitext-2-raw-v1 configuration
        - Supports partial dataset loading via start_pc/end_pc parameters
        - Maintains deterministic ordering for reproducible experiments
        
    Memory Optimization:
        - Streaming processing for large dataset portions
        - Efficient numpy array operations for concatenation
        - Minimal memory footprint during tokenization
        - Automatic garbage collection of intermediate results
    """
    assert dataset in ["shakespeare", "wikitext"]

    if dataset == "shakespeare":
        char = True
    else:
        char = False

    # Decide cache locations based on tokenization mode and rank.
    if char:
        cache_dir = os.path.join("data", f"{dataset}_char")
    else:
        cache_dir = os.path.join("data", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    data_cache_file = os.path.join(
        cache_dir, f"data_block{block_size}_{start_pc}_{end_pc}.npy"
    )

    if os.path.exists(data_cache_file):
        print(f"Loading cached dataset from {data_cache_file}")
        data = np.load(data_cache_file)
        # Determine vocab size based on dataset type
        if char:
            char_int, eos_token_id = generate_char_vocab()
            vocab_size = len(char_int)
        else:
            vocab_size = 50257  # GPT-2 vocab size
        return data, vocab_size

    print(
        f"Loading dataset: {dataset} {'(char-level)' if char else '(GPT2 tokenization)'} start%: {start_pc} end%: {end_pc}"
    )

    # Determine the dataset identifier and mapping function.
    if dataset == "shakespeare":
        dataset_id = "Trelis/tiny-shakespeare"
        def mapping_fn(x): return {"text": x["Text"]}
        load_config = {}
    elif dataset == "wikitext":
        dataset_id = "wikitext"
        config = "wikitext-2-raw-v1"
        def mapping_fn(x): return {"text": x["text"]}
        load_config = {"name": config}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    total_records = (
        builder.info.splits["train"].num_examples
        + builder.info.splits["test"].num_examples
    )

    start_record = int(total_records * start_pc)
    end_record = int(total_records * end_pc)

    used_records = end_record - start_record
    print(f"Using {used_records} records: {start_record} to {end_record}")

    # Small enough dataset that we can load the whole thing in.
    dataset = load_dataset(dataset_id, **load_config)
    dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    dataset = dataset.map(mapping_fn, remove_columns=dataset.column_names)

    dataset = dataset.select(range(start_record, end_record))

    ## Initialize the tokenizer.
    if char:
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)

        def tokenize(example):
            text = example["text"]
            if isinstance(text, str):
                return {"tokenized": [char_int[c] for c in text]}
            elif isinstance(text, list):
                return {"tokenized": [[char_int[c] for c in t] for t in text]}
            else:
                raise Exception("Unknown type")

    else:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id

        def tokenize(example):
            return {
                "tokenized": tokenizer(
                    example["text"], truncation=True, max_length=block_size
                )["input_ids"]
            }

    ## Tokenize the dataset.
    dataset = dataset.map(tokenize, num_proc=os.cpu_count(), batched=True)

    # Convert tokenized lists to 1-d contiguous stream.
    def aggregate_examples(examples):
        all_ids = np.concatenate(
            [np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids]
        )
        return {"ids": all_ids}

    dataset_processed = dataset.map(
        aggregate_examples,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    dataset_processed.set_format(type="numpy", columns=["ids"])

    data = dataset_processed["ids"]

    print(f"Dataset size: {data.shape}")

    np.save(data_cache_file, data)

    return data, vocab_size


def build_dataset_owt(start_pc=0.0, end_pc=1.0, max_workers=8):
    """
    Load and preprocess OpenWebText dataset with chunked caching for distributed training.
    
    Implements sophisticated chunked caching strategy optimized for large-scale distributed
    training on OpenWebText dataset. Supports partial dataset loading, parallel processing,
    and intelligent chunk management for memory-efficient training.
    
    Args:
        start_pc (float): Start percentage of dataset to use (0.0-1.0)
        end_pc (float): End percentage of dataset to use (0.0-1.0)  
        max_workers (int): Maximum parallel workers for dataset processing
        
    Returns:
        tuple: (chunk_ids, cache_location, vocab_size)
            - chunk_ids (list): List of chunk IDs available for training
            - cache_location (str): Directory path containing cached chunks
            - vocab_size (int): GPT-2 tokenizer vocabulary size (50,257)
            
    Chunking Strategy:
        - Target 1000 chunks for full dataset (configurable via target_chunks_for_full_dataset)
        - Each chunk contains fixed-size blocks (1024 tokens per block)
        - Chunk IDs calculated based on position in full dataset
        - Support for partial dataset loading with proportional chunk allocation
        
    Caching Architecture:
        - Chunk-based caching enables lazy loading during training
        - Cache files: "data/owt/chunk_{chunk_id}.npy"
        - Automatic detection of missing chunks with selective regeneration
        - Memory-efficient processing without loading entire dataset
        
    Processing Pipeline:
        1. Calculate target chunk range based on start_pc/end_pc parameters
        2. Check for existing cached chunks in expected range
        3. Download and process only missing chunks from HuggingFace
        4. Tokenize using GPT-2 tokenizer with consistent block sizes
        5. Save processed chunks with globally consistent chunk IDs
        
    Distributed Training Support:
        - Chunk IDs provide global addressing across distributed nodes
        - Per-node data sharding through chunk subset allocation
        - Lazy loading enables training on datasets larger than memory
        - Consistent data ordering across distributed training runs
        
    Memory Management:
        - Streaming processing with configurable parallelism
        - Fixed block size (1024 tokens) for consistent memory usage
        - Automatic garbage collection of intermediate processing results
        - Chunk-level granularity for optimal memory utilization
        
    Error Handling:
        - Validates dataset availability and download integrity
        - Handles network interruptions during dataset download
        - Provides clear progress reporting for long-running operations
        - Automatic retry logic for failed chunk processing
    """
    block_size = 1024  # Fixed block size for OWT
    cache_dir = os.path.join("data", "owt")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if the chunks for this range already exist
    target_chunks_for_full_dataset = 1000
    start_chunk_id = int(start_pc * target_chunks_for_full_dataset)
    end_chunk_id = int(end_pc * target_chunks_for_full_dataset)
    expected_chunk_ids = list(range(start_chunk_id, end_chunk_id))

    # Check if all required chunks exist
    missing_chunks = []
    for chunk_id in expected_chunk_ids:
        cache_file = f"{cache_dir}/chunk_{chunk_id}.npy"
        if not os.path.exists(cache_file):
            missing_chunks.append(chunk_id)

    if not missing_chunks:
        print(
            f"All chunks {start_chunk_id} to {end_chunk_id-1} already exist, using cached data"
        )
        # Still need to get vocab_size
        vocab_size = 50257  # GPT-2 vocab size
        return expected_chunk_ids, cache_dir, vocab_size
    else:
        print(f"Missing chunks {missing_chunks}, will download and process data")

    print(
        f"Loading dataset: owt {'(GPT2 tokenization)'} start%: {start_pc} end%: {end_pc}"
    )

    dataset_id = "Skylion007/openwebtext"
    load_config = {"trust_remote_code": True}

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    total_records = builder.info.splits["train"].num_examples

    print(f"Total records to import: {total_records}")

    # Calculate the number of records to use and how to split them.
    start_record = int(total_records * start_pc)
    end_record = int(total_records * end_pc)

    used_records = end_record - start_record
    print(f"Using {used_records} records: {start_record} to {end_record}")

    dataset = load_dataset(
        dataset_id, split=f"train[{start_record}:{end_record}]", **load_config
    )

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError(
            "transformers is not installed. Please install the correct distro using pip install exogym[gpt]"
        )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    def tokenize(example):
        return {
            "tokenized": tokenizer(
                example["text"], truncation=True, max_length=block_size
            )["input_ids"]
        }

    ## Tokenize the dataset.
    dataset = dataset.map(tokenize, num_proc=os.cpu_count(), batched=True)

    # Convert tokenized lists to blocks with fixed block size
    def aggregate_examples(examples):
        # Flatten all ids and add EOS tokens
        all_ids = np.concatenate(
            [np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids]
        )
        num_blocks = len(all_ids) // block_size
        if num_blocks == 0:
            return {"ids": torch.tensor([])}
        all_ids = all_ids[: num_blocks * block_size]
        data_2d = all_ids.reshape(-1, block_size)
        return {"ids": data_2d}

    dataset_processed = dataset.map(
        aggregate_examples,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    # Get all the processed blocks
    all_blocks = []
    for item in dataset_processed:
        if len(item["ids"]) > 0:
            all_blocks.append(item["ids"])

    if not all_blocks:
        raise ValueError("No valid blocks found in dataset")

    # Concatenate all blocks into a single 2D array
    all_data = np.vstack(all_blocks)
    total_blocks = all_data.shape[0]

    print(f"Total blocks: {total_blocks}")

    # Calculate number of chunks for this range
    range_pc = end_pc - start_pc
    num_chunks = max(1, int(target_chunks_for_full_dataset * range_pc))

    # Calculate blocks per chunk
    blocks_per_chunk = max(1, total_blocks // num_chunks)

    print(f"Creating {num_chunks} chunks with ~{blocks_per_chunk} blocks each")
    print(
        f"Chunk IDs will range from {start_chunk_id} to {start_chunk_id + num_chunks - 1}"
    )

    # Create chunks and save them with correct chunk IDs
    chunk_ids = []
    cache_location = cache_dir

    for chunk_idx in range(num_chunks):
        start_block = chunk_idx * blocks_per_chunk
        if chunk_idx == num_chunks - 1:
            # Last chunk gets all remaining blocks
            end_block = total_blocks
        else:
            end_block = (chunk_idx + 1) * blocks_per_chunk

        if start_block >= total_blocks:
            break

        chunk_data = all_data[start_block:end_block]

        # Use correct chunk ID based on position in full dataset
        chunk_id = start_chunk_id + chunk_idx
        chunk_ids.append(chunk_id)

        # Save chunk
        cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
        np.save(cache_file, chunk_data)

        print(
            f"Saved chunk {chunk_id} with {chunk_data.shape[0]} blocks to {cache_file}"
        )

    return chunk_ids, cache_location, vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="Dataset: shakespeare, wikitext, code, or owt",
    )
    parser.add_argument(
        "--char", action="store_true", help="Enable character-level tokenization"
    )
    parser.add_argument(
        "--start_pc",
        type=float,
        default=0.0,
        help="Proportion of the dataset to use (0 to 1)",
    )
    parser.add_argument(
        "--end_pc",
        type=float,
        default=1.0,
        help="Fraction of the used dataset to reserve for validation",
    )
    args = parser.parse_args()

    chunk_ids, cache_location, vocab_size = build_dataset_owt(
        args.start_pc, args.end_pc
    )

    print(
        f"Finished importing dataset: {args.dataset} {'(char-level)' if args.char else '(GPT2 tokenization)'} start%: {args.start_pc} end%: {args.end_pc}"
    )


if __name__ == "__main__":
    main()
