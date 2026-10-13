import pandas as pd
import os

# Locate the dataset file path
csv_path = os.path.join("data", "hospital_readmission_dataset.csv")

try:
    df = pd.read_csv(csv_path)
    print("✅ DATA EXTRACTION SUCCESSFUL!")
    print(f"Total Operational Records: {df.shape[0]} rows")
    print(f"Total Feature Dimensions: {df.shape[1]} columns\n")
    print("--- COLUMN SCHEMAS AVAILABLE TO THE AGENT ---")
    print(df.dtypes)
    print("\n--- FIRST 2 SAMPLE RECORDS ---")
    print(df.head(2))
except Exception as e:
    print(f"❌ Failed to read data file. Verify file name in data/ directory. Error: {e}")