
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"

df = pd.read_csv(DATA_PATH)

# Convert the date column for time-based analysis
df["Date"] = pd.to_datetime(
    df["Date"], format="%Y-%m-%d", errors="raise"
)

amount_column = "Transaction Amount"

print("\n--- OVERALL TRANSACTION STATISTICS ---")
print(df[amount_column].describe().round(2))

print("\n--- CATEGORY-WISE SPENDING ---")
category_spending = (
    df.groupby("Category")[amount_column]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)

category_spending.columns = [
    "Transaction Count", "Total Amount", "Average Amount"
]

print(category_spending.round(2))

print("\n--- MONTHLY TRANSACTION SUMMARY ---")
df["Month"] = df["Date"].dt.to_period("M")

monthly_spending = (
    df.groupby("Month")[amount_column]
    .agg(["count", "sum", "mean"])
)

monthly_spending.columns = [
    "Transaction Count", "Total Amount", "Average Amount"
]

print(monthly_spending.round(2))

print("\nEDA summary completed!")
