#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path


def check_notebooks(directory):
    """
    Executes all Jupyter notebooks in a directory to check for errors.

    Args:
        directory (str): The path to the directory containing notebooks.

    Returns:
        bool: True if all notebooks run successfully, False otherwise.
    """
    notebook_dir = Path(directory)
    if not notebook_dir.is_dir():
        print(
            f"Info: Directory '{directory}' not found, skipping notebook check.", file=sys.stderr)
        return True

    notebooks = list(notebook_dir.glob('*.ipynb'))
    if not notebooks:
        print(f"Info: No notebooks found in '{directory}'.", file=sys.stderr)
        return True

    all_passed = True
    for notebook in notebooks:
        print(f"Checking notebook: {notebook}...")
        try:
            # Execute the notebook using nbconvert
            # The output is discarded, we only care about the exit code
            subprocess.run(
                [
                    sys.executable, '-m', 'jupyter', 'nbconvert',
                    '--to', 'notebook', '--execute',
                    '--stdout', str(notebook)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  \u2713 Success: {notebook} executed without errors.")
        except subprocess.CalledProcessError as e:
            print(
                f"  \u2717 Error: Failed to execute notebook: {notebook}", file=sys.stderr)
            print("\n" + "="*80, file=sys.stderr)
            print(f"ERROR in {notebook}", file=sys.stderr)
            print("="*80, file=sys.stderr)
            # Print the error output from nbconvert
            print(e.stderr, file=sys.stderr)
            all_passed = False
        except FileNotFoundError:
            print("Error: 'jupyter' command not found.", file=sys.stderr)
            print("Please install Jupyter: pip install jupyter", file=sys.stderr)
            return False

    return all_passed


def main():
    """
    Main function to run the notebook check.
    """
    if check_notebooks('notebooks'):
        print("\nAll notebooks executed successfully!")
        sys.exit(0)
    else:
        print("\nOne or more notebooks failed to execute.")
        sys.exit(1)


if __name__ == "__main__":
    main()
