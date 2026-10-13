
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(
    df["Date"], format="%Y-%m-%d", errors="raise"
)

amount = "Transaction Amount"

# Chart 1: Category-wise transaction value
category_totals = (
    df.groupby("Category")[amount]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
category_totals.plot(kind="bar")
plt.title("Total Transaction Amount by Category")
plt.xlabel("Category")
plt.ylabel("Total Transaction Amount")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_spending.png", dpi=150)
plt.show()
plt.close()

# Chart 2: Monthly transaction value
monthly_totals = (
    df.set_index("Date")[amount]
    .resample("MS")
    .sum()
)

plt.figure(figsize=(9, 5))
monthly_totals.plot(kind="line", marker="o")
plt.title("Monthly Transaction Amount")
plt.xlabel("Month")
plt.ylabel("Total Transaction Amount")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_transactions.png", dpi=150)
plt.show()
plt.close()

print("Charts generated successfully!")
print(f"Saved to: {OUTPUT_DIR}")
