#!/usr/bin/env python
"""
Dataset Download Utility - Standalone Dataset Preprocessing

This utility script provides a standalone interface for downloading and preprocessing
datasets without running actual training. Useful for pre-populating dataset caches,
CI/CD pipeline setup, and distributed training preparation where dataset preprocessing
should be separated from training execution.

Role in System:
- Standalone dataset preprocessing and caching utility
- CI/CD pipeline component for dataset preparation
- Development tool for pre-populating caches before training experiments
- Distributed training preparation tool for ensuring consistent dataset availability

Called by:
- CI/CD systems preparing datasets for distributed training experiments
- Developers setting up local development environments
- System administrators preparing distributed training infrastructure
- Manual execution: `python download_dataset.py --dataset owt --proportion 0.1`

Calls:
- nanogpt/dataset.py get_dataset() for unified dataset processing interface
- argparse for command-line interface and parameter validation
- Underlying dataset processing pipeline through dataset factory

Usage Patterns:
- Full dataset download: `python download_dataset.py --dataset owt`
- Partial dataset for testing: `python download_dataset.py --dataset owt --proportion 0.01`
- Multi-threaded processing: `python download_dataset.py --max_workers 16`
- Educational datasets: `python download_dataset.py --dataset shakespeare`

Design Considerations:
- Separates dataset preparation from training for cleaner workflows
- Provides progress feedback for long-running dataset downloads
- Supports configurable parallelism for optimal download performance
- Validates dataset integrity before completing
"""

import argparse
from dataset import get_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset files for DistributedSim"
    )
    parser.add_argument(
        "--dataset", type=str, default="owt", help="Dataset identifier (default: owt)"
    )
    parser.add_argument(
        "--proportion",
        type=float,
        default=1.0,
        help="Proportion of dataset to download (0.0-1.0)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of threads to use for downloading (default: 8)",
    )

    args = parser.parse_args()

    # Call get_dataset which will download and cache the data
    # We discard the returned dataset since we only want to download
    _, _ = get_dataset(args.dataset, 0, args.proportion, max_workers=args.max_workers)

    print("Dataset download complete!")


if __name__ == "__main__":
    main()
