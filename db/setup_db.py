from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
csv_path = PROJECT_ROOT / "data" / "hospital_readmission_dataset.csv"
db_path = PROJECT_ROOT / "db" / "healthcare_data.db"

print("📦 Initializing Enterprise Data Ingestion for Business Analyst Schema...")

if not csv_path.exists():
    print(f"❌ Target CSV file not found at: {csv_path}")
    print("👉 Please make sure 'hospital_readmission_dataset.csv' is placed inside your 'data/' folder.")
else:
    # Read the Kaggle source data
    df = pd.read_csv(csv_path)
    
    # Standardize column strings for clean SQL execution (lowercase, no spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Establish connection and write table
    conn = sqlite3.connect(db_path)
    df.to_sql("readmissions", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"✅ Production SQL Database updated at: {db_path}")
    print("   Target Table: 'readmissions'")
    print(f"   Indexed Operational Records: {len(df)} rows ready for live queries.")
