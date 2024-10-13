#!/usr/bin/env python3

import sys
from datasets import load_from_disk
from tabulate import tabulate

def read_dataset(directory):
    try:
        dataset = load_from_disk(directory)
        return dataset
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

def display_dataset(dataset):
    # Display the first few rows
    print(tabulate(dataset[:5], headers='keys', tablefmt="grid"))

    # Print additional information
    print(f"\nNumber of rows: {len(dataset)}")
    print(f"Number of columns: {len(dataset.column_names)}")

    # Print features (schema)
    print("\nFeatures:")
    print(dataset.features)

    # Print column names
    print("\nColumn names:")
    print(", ".join(dataset.column_names))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_hf_dataset.py <dataset_directory>")
        sys.exit(1)

    directory = sys.argv[1]
    dataset = read_dataset(directory)
    display_dataset(dataset)
