#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import argparse

def clear_notebook_outputs(root_dir='.'):
    """Find all Jupyter notebooks and clear their outputs using nbconvert."""
    # Find all .ipynb files recursively
    notebook_files = list(Path(root_dir).rglob('*.ipynb'))
    
    count = 0
    for nb_path in notebook_files:
        print(f"Processing: {nb_path}")
        try:
            # Run jupyter nbconvert command
            subprocess.run(['jupyter', 'nbconvert', '--clear-output', '--inplace', str(nb_path)], 
                          check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error processing {nb_path}: {e}")
    
    print(f"Completed! Processed {count} notebooks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear outputs from Jupyter notebooks recursively')
    parser.add_argument('--dir', type=str, default='.', help='Root directory to search for notebooks (default: current directory)')
    
    args = parser.parse_args()
    clear_notebook_outputs(args.dir)
