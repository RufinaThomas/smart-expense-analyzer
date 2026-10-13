# Smart Expense & Spending Analyzer

A Python-based data analytics project that explores transaction patterns, analyzes category-wise spending, visualizes monthly trends, and predicts transaction categories using machine learning.

## Features

- **Spending Overview:** Total transaction value, average transaction value, highest transaction value, and unique merchants.
- **Category Analysis:** Category-wise transaction values, distribution charts, and summary tables.
- **Monthly Trends:** Monthly transaction values and transaction counts.
- **Transaction Explorer:** Search, sort, filter, and download transaction records.
- **Machine Learning:** Predict a likely transaction category from a transaction amount.
- **Model Evaluation:** View held-out test accuracy, Macro-F1 score, and confusion matrix.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning

The project uses a Logistic Regression pipeline with feature scaling to predict transaction categories from transaction amounts.

| Metric | Result |
|---|---:|
| Test accuracy | 48.06% |
| Macro-F1 | 0.472 |

The model was evaluated on 10,000 held-out test records after training on 40,000 records.

**Limitation:** The model uses transaction amount alone. Its predictions are estimates, and its performance may not generalize to other datasets. Category-level performance varies.

## Project Structure

```text
smart-expense-analyzer/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── credit_card_transaction_flow.csv
│   └── cleaned_transactions.csv
├── src/
│   ├── clean_data.py
│   ├── eda_analysis.py
│   ├── eda_visualizations.py
│   └── train_model.py
├── reports/
│   ├── figures/
│   ├── models/
│   │   └── merchant_category_model.joblib
│   ├── model_metrics.csv
│   └── confusion_matrix.csv
└── tests/
```

## Installation

Clone the repository and open the project directory:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd smart-expense-analyzer
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

## Data and Privacy

The application expects a cleaned transaction dataset at `data/cleaned_transactions.csv` and a trained model at `reports/models/merchant_category_model.joblib`.

Before distributing the project, verify the dataset's license and remove personal or identifying information where necessary. Do not publish customer names, IDs, birthdates, or other sensitive data without authorization.

## Future Improvements

- Improve category prediction using additional appropriate features.
- Compare multiple classification algorithms.
- Add automated model evaluation and validation.
- Add unit tests for data cleaning and prediction.
- Support safe uploads of user-provided transaction files.

## Disclaimer

This is an educational data analytics project. Model predictions are not guaranteed to be correct, and the results depend on the dataset used.
