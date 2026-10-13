from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ml_transactions.csv"

FEATURE_COLUMNS = ["Transaction Amount"]
TARGET_COLUMN = "Category"


def load_cleaned_transactions(path: Path) -> pd.DataFrame:
    """Load the cleaned transaction data and validate ML columns."""
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {path}")

    df = pd.read_csv(path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def prepare_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready rows with numeric features and labels."""
    ml_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    ml_df["Transaction Amount"] = pd.to_numeric(
        ml_df["Transaction Amount"],
        errors="coerce",
    )
    ml_df[TARGET_COLUMN] = ml_df[TARGET_COLUMN].astype("string").str.strip()

    ml_df = ml_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    ml_df = ml_df[ml_df["Transaction Amount"] >= 0]

    if ml_df.empty:
        raise ValueError("No valid ML rows remain after preparation.")

    return ml_df


def main() -> None:
    df = load_cleaned_transactions(INPUT_PATH)
    ml_df = prepare_ml_data(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUTPUT_PATH, index=False)

    print("ML data preparation completed!")
    print(f"Rows saved: {len(ml_df):,}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
