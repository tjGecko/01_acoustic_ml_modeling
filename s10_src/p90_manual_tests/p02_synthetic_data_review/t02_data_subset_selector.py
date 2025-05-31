#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to find and copy files with a specific prefix to a target directory.

SETUP INSTRUCTIONS:
1. Create and activate a virtual environment (recommended):
   ```bash
   # From the project root directory:
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # OR
   .\venv\Scripts\activate  # On Windows
   
   # Install required packages
   pip install -r requirements.txt
   pip install tqdm  # If not in requirements.txt
   ```

2. Run the script from the project root directory:
   ```bash
   python s10_src/p90_manual_tests/p02_synthetic_data_review/t02_data_subset_selector.py
   ```

3. Verify the source and target directories in the script if needed.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # Source directory and prefix
    source_dir = Path("/media/tj/Samsung_T5/Ziz/01_time_domain/capture_2025_05_29/data")
    prefix = "mixed_46-bebop_001__"
    
    # Target directory
    target_dir = Path("/home/tj/99_tmp/11 - synthetic mic array data/01_time_domain/")
    
    print(f"Searching for files with prefix: {prefix}")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    # Ensure source directory exists
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files with the prefix
    matching_files = list(source_dir.glob(f"{prefix}*"))
    
    if not matching_files:
        print(f"No files found with prefix: {prefix}")
        return
    
    print(f"Found {len(matching_files)} matching files.")
    
    # Copy each file to the target directory with progress bar
    print("\nCopying files...")
    copied_count = 0
    
    with tqdm(total=len(matching_files), unit='file') as pbar:
        for src_path in matching_files:
            if not src_path.is_file():
                print(f"\nSkipping non-file: {src_path}")
                pbar.update(1)
                continue
                
            dst_path = target_dir / src_path.name
            pbar.set_description(f"Copying: {src_path.name[:20]}...")
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"\nError copying {src_path.name}: {str(e)}")
            
            pbar.update(1)
    
    print("\nCopy operation completed!")
    print(f"Successfully copied {copied_count} of {len(matching_files)} files to {target_dir}")
    if copied_count < len(matching_files):
        print(f"Note: {len(matching_files) - copied_count} files were not copied due to errors.")

if __name__ == "__main__":
    main()