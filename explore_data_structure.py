# explore_data_structure.py
import pandas as pd
import numpy as np
from pathlib import Path

def explore_data():
    """Explore the data structure before hypothesis testing."""
    
    file_path = Path("data/raw/machineLearningRating_v3.txt")
    
    # First, check if it's CSV or another format
    print("Checking file format...")
    
    # Read first few lines to understand structure
    with open(file_path, 'r') as f:
        first_lines = [f.readline().strip() for _ in range(5)]
    
    print("\nFirst 5 lines of the file:")
    for i, line in enumerate(first_lines, 1):
        print(f"{i}: {line[:100]}..." if len(line) > 100 else f"{i}: {line}")
    
    # Try to determine delimiter
    delimiters = [',', ';', '\t', '|']
    for delim in delimiters:
        if delim in first_lines[0]:
            print(f"\nLikely delimiter: '{delim}'")
            # Try to load as DataFrame
            try:
                df = pd.read_csv(file_path, delimiter=delim, nrows=1000)
                print(f"Successfully loaded with '{delim}' delimiter")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                # Check for key columns needed for hypothesis testing
                required_cols = ['Province', 'PostalCode', 'Gender', 'TotalPremium', 'TotalClaims']
                available_cols = [col for col in required_cols if col in df.columns]
                print(f"\nAvailable key columns for hypothesis testing: {available_cols}")
                
                # Show sample
                print("\nFirst few rows:")
                print(df.head())
                return df, delim
                
            except Exception as e:
                print(f"Failed with delimiter '{delim}': {e}")
    
    print("\nCould not determine file format automatically.")
    return None, None

if __name__ == "__main__":
    explore_data()