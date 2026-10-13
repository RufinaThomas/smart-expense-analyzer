from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"


def load_transactions(path: Path) -> pd.DataFrame:
    """Load transaction data and validate merchant/category columns."""
    if not path.exists():
        raise FileNotFoundError(f"Transaction data file not found: {path}")

    df = pd.read_csv(path)
    required_columns = ["Merchant Name", "Category", "Transaction Amount"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def summarize_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated merchants and category consistency."""
    work_df = df.copy()
    work_df["Merchant Name"] = (
        work_df["Merchant Name"].astype("string").str.strip()
    )
    work_df["Category"] = work_df["Category"].astype("string").str.strip()
    work_df["Transaction Amount"] = pd.to_numeric(
        work_df["Transaction Amount"],
        errors="coerce",
    )

    work_df = work_df.dropna(
        subset=["Merchant Name", "Category", "Transaction Amount"]
    )

    summary = (
        work_df.groupby("Merchant Name")
        .agg(
            Transaction_Count=("Merchant Name", "size"),
            Category_Count=("Category", "nunique"),
            Most_Common_Category=(
                "Category",
                lambda values: values.mode().iat[0],
            ),
            Total_Amount=("Transaction Amount", "sum"),
            Average_Amount=("Transaction Amount", "mean"),
        )
        .reset_index()
        .sort_values(
            ["Transaction_Count", "Total_Amount"],
            ascending=[False, False],
        )
    )

    return summary


def main() -> None:
    df = load_transactions(DATA_PATH)
    merchant_summary = summarize_merchants(df)
    repeated_merchants = merchant_summary[
        merchant_summary["Transaction_Count"] > 1
    ]

    print("\n--- MERCHANT REPETITION SUMMARY ---")
    print(f"Total merchants: {len(merchant_summary):,}")
    print(f"Repeated merchants: {len(repeated_merchants):,}")

    print("\n--- TOP REPEATED MERCHANTS ---")
    print(repeated_merchants.head(20).round(2).to_string(index=False))

    inconsistent_categories = repeated_merchants[
        repeated_merchants["Category_Count"] > 1
    ]

    print("\n--- MERCHANTS APPEARING IN MULTIPLE CATEGORIES ---")
    if inconsistent_categories.empty:
        print("No repeated merchants appear in multiple categories.")
    else:
        print(
            inconsistent_categories.head(20).round(2).to_string(index=False)
        )


if __name__ == "__main__":
    main()
