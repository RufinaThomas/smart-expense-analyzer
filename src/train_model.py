
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "reports" / "models"
MODEL_PATH = MODEL_DIR / "merchant_category_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.csv"
CONFUSION_PATH = PROJECT_ROOT / "reports" / "confusion_matrix.csv"

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

df = df[
    ["Transaction Amount", "Category"]
].dropna()

X = df[["Transaction Amount"]]
y = df["Category"]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Build the selected model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# 5. Train the model
print("Training the amount-only model...")
model.fit(X_train, y_train)

# 6. Evaluate on the held-out test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

print("\n--- FINAL MODEL EVALUATION ---")
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

# 7. Create output folders
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 8. Save the trained model
joblib.dump(model, MODEL_PATH)

# 9. Save summary metrics
metrics_df = pd.DataFrame([
    {
        "Model": "Transaction amount only",
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Training Records": len(X_train),
        "Testing Records": len(X_test),
    }
])

metrics_df.to_csv(METRICS_PATH, index=False)

# 10. Save the confusion matrix
labels = sorted(y.unique())

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

confusion_df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
)

confusion_df.index.name = "Actual Category"
confusion_df.to_csv(CONFUSION_PATH)

# 11. Confirm saved files
print("\n--- SAVED FILES ---")
print(f"Model: {MODEL_PATH}")
print(f"Metrics: {METRICS_PATH}")
print(f"Confusion matrix: {CONFUSION_PATH}")

print("\nTraining, evaluation, and saving completed!")
