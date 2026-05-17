#!/usr/bin/env python
import os
import hashlib
import sys
from collections import defaultdict

def find_duplicate_files(directory):
    """
    Finds duplicate files in a given directory based on their content hash.
    
    Args:
        directory (str): The path to the directory to scan.
        
    Returns:
        dict: A dictionary mapping each file hash to a list of paths of duplicate files.
              Returns an empty dictionary if no duplicates are found.
    """
    hashes = defaultdict(list)
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                hashes[file_hash].append(filepath)
            except (IOError, OSError) as e:
                print(f"Warning: Could not read file {filepath}: {e}", file=sys.stderr)

    return {hash_val: paths for hash_val, paths in hashes.items() if len(paths) > 1}

def main():
    """
    Main function to check for duplicates in the 'data/' directory and exit.
    """
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        print(f"Info: Directory '{data_dir}' not found, skipping duplicate check.", file=sys.stderr)
        sys.exit(0)

    duplicates = find_duplicate_files(data_dir)

    if duplicates:
        print("Error: Duplicate files found in 'data/' directory!", file=sys.stderr)
        for hash_val, paths in duplicates.items():
            print(f"\nHash: {hash_val}", file=sys.stderr)
            for path in paths:
                print(f"  - {path}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Success: No duplicate files found in 'data/' directory.", file=sys.stdout)
        sys.exit(0)

if __name__ == "__main__":
    main()
