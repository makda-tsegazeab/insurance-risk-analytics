# create_data_documentation.py
import pandas as pd
import os

# Get file info
file_path = "data/raw/machineLearningRating_v3.txt"
file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB

print(f"File: {file_path}")
print(f"Size: {file_size:.2f} GB")
print(f"Location: data/raw/")

# If possible, read just the header
try:
    # Try to read first few lines to understand structure
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        print(f"\nHeader (first line): {header[:100]}...")
        
        # Count approximate number of lines
        print("\nCounting lines...")
        line_count = sum(1 for _ in open(file_path))
        print(f"Total lines: {line_count:,}")
        
except Exception as e:
    print(f"Could not read file: {e}")