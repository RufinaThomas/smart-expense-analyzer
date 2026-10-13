from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the minimum dataset needed for a baseline classifier."""
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    df = pd.read_csv(path)
    required_columns = ["Transaction Amount", "Category"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].dropna().copy()
    df["Transaction Amount"] = pd.to_numeric(
        df["Transaction Amount"],
        errors="coerce",
    )
    df = df.dropna(subset=required_columns)

    if df.empty:
        raise ValueError("No valid rows available for baseline modeling.")

    return df[["Transaction Amount"]], df["Category"]


def main() -> None:
    X, y = load_training_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    predictions = baseline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\n--- BASELINE MODEL EVALUATION ---")
    print("Strategy: predict the most frequent category")
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
