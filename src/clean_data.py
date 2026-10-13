
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "credit_card_transaction_flow.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"

# Load the original dataset
df = pd.read_csv(DATA_PATH)

# Convert date columns to datetime
df["Date"] = pd.to_datetime(
    df["Date"], format="%d-%m-%Y", errors="raise"
)

df["Birthdate"] = pd.to_datetime(
    df["Birthdate"], format="%d-%m-%Y", errors="raise"
)

# Remove accidental whitespace from text columns
text_columns = [
    "Name", "Surname", "Gender", "Merchant Name", "Category"
]

for column in text_columns:
    df[column] = df[column].str.strip()

# Standardize category capitalization
df["Category"] = df["Category"].str.title()

# Check for missing required fields
required_columns = [
    "Customer ID", "Transaction Amount", "Date",
    "Merchant Name", "Category"
]

if df[required_columns].isna().any().any():
    raise ValueError("Missing values found in required columns.")

# Save the cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("Data cleaning completed!")
print(f"Rows saved: {len(df)}")
print(f"Output file: {OUTPUT_PATH}")
print("\nCategories:")
print(df["Category"].value_counts())
