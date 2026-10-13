from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Expense Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PATHS AND DATA LOADING
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "cleaned_transactions.csv"

MODEL_PATH = (
    PROJECT_DIR
    / "reports"
    / "models"
    / "merchant_category_model.joblib"
)

METRICS_PATH = PROJECT_DIR / "reports" / "model_metrics.csv"
CONFUSION_PATH = PROJECT_DIR / "reports" / "confusion_matrix.csv"


@st.cache_resource
def load_model(model_path):
    """Load the trained machine-learning model."""
    return joblib.load(model_path)


@st.cache_data
def load_data(file_path):
    """Load and validate the cleaned transaction dataset."""

    df = pd.read_csv(file_path)

    required_columns = [
        "Transaction Amount",
        "Date",
        "Category",
        "Merchant Name",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    df["Transaction Amount"] = pd.to_numeric(
        df["Transaction Amount"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Transaction Amount",
            "Category",
        ]
    ).copy()

    df = df[df["Transaction Amount"] >= 0].copy()

    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    return df


# =========================================================
# 3. LOAD DATA
# =========================================================

st.title("💳 Smart Expense Analyzer")

st.caption(
    "Analyze transaction patterns, compare categories, "
    "and explore spending trends."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Make sure cleaned_transactions.csv is inside the data folder."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except (ValueError, OSError, pd.errors.ParserError) as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.warning("No valid transactions are available for analysis.")
    st.stop()

st.success(
    f"Dataset loaded successfully: {len(df):,} valid transactions."
)


# =========================================================
# 4. SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Filter Transactions")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Transaction date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

all_categories = sorted(
    df["Category"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Categories",
    options=all_categories,
    default=all_categories,
)

merchant_options = sorted(
    df["Merchant Name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_merchants = st.sidebar.multiselect(
    "Merchants",
    options=merchant_options,
    help="Leave empty to include all merchants.",
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

if start_date > end_date:
    st.sidebar.error("Start date must not be after end date.")
    st.stop()

filtered_df = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
    & (df["Category"].isin(selected_categories))
].copy()

if selected_merchants:
    filtered_df = filtered_df[
        filtered_df["Merchant Name"]
        .astype(str)
        .isin(selected_merchants)
    ]

st.sidebar.caption(
    f"{len(filtered_df):,} transactions match your filters."
)


# =========================================================
# 5. KEY PERFORMANCE INDICATORS
# =========================================================

st.header("Spending Overview")

if filtered_df.empty:
    st.warning(
        "No transactions match the selected filters. "
        "Change the filters in the sidebar."
    )
    st.stop()

total_value = filtered_df["Transaction Amount"].sum()
average_value = filtered_df["Transaction Amount"].mean()
highest_value = filtered_df["Transaction Amount"].max()
unique_merchants = filtered_df["Merchant Name"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Transaction Value",
        f"{total_value:,.2f}",
    )

with col2:
    st.metric(
        "Average Transaction Value",
        f"{average_value:,.2f}",
    )

with col3:
    st.metric(
        "Highest Transaction Value",
        f"{highest_value:,.2f}",
    )

with col4:
    st.metric(
        "Unique Merchants",
        f"{unique_merchants:,}",
    )

st.caption(
    f"Based on {len(filtered_df):,} transactions across "
    f"{filtered_df['Category'].nunique()} categories. "
    "Amounts are shown without a currency symbol because "
    "the dataset's currency has not been confirmed."
)


# =========================================================
# 6. CATEGORY-WISE SPENDING
# =========================================================

st.header("Category Analysis")

category_summary = (
    filtered_df.groupby("Category")["Transaction Amount"]
    .agg(["sum", "mean", "count"])
    .reset_index()
    .rename(
        columns={
            "sum": "Total Value",
            "mean": "Average Value",
            "count": "Transaction Count",
        }
    )
    .sort_values("Total Value", ascending=False)
)

left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("Total Value by Category")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        category_summary["Category"],
        category_summary["Total Value"],
    )

    ax.set_xlabel("Total Transaction Value")
    ax.set_ylabel("Category")
    ax.set_title("Category-wise Total Value")
    ax.invert_yaxis()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right_chart:
    st.subheader("Share of Total Value")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.pie(
        category_summary["Total Value"],
        labels=category_summary["Category"],
        autopct="%1.1f%%",
        startangle=90,
    )

    ax.set_title("Transaction Value Distribution")
    ax.axis("equal")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Category Summary Table")

display_category = category_summary.copy()

display_category["Total Value"] = display_category[
    "Total Value"
].round(2)

display_category["Average Value"] = display_category[
    "Average Value"
].round(2)

st.dataframe(
    display_category,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 7. MONTHLY TRENDS
# =========================================================

st.header("Monthly Trends")

monthly_summary = (
    filtered_df.groupby("Month")
    .agg(
        Total_Value=("Transaction Amount", "sum"),
        Transaction_Count=("Transaction Amount", "count"),
        Average_Value=("Transaction Amount", "mean"),
    )
    .reset_index()
    .sort_values("Month")
)

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Monthly Transaction Value")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        monthly_summary["Month"],
        monthly_summary["Total_Value"],
        marker="o",
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Total Transaction Value")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with trend_right:
    st.subheader("Monthly Transaction Count")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        monthly_summary["Month"],
        monthly_summary["Transaction_Count"],
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Transactions")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Monthly Summary Table")

display_monthly = monthly_summary.copy()

display_monthly["Total_Value"] = display_monthly[
    "Total_Value"
].round(2)

display_monthly["Average_Value"] = display_monthly[
    "Average_Value"
].round(2)

st.dataframe(
    display_monthly,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 8. TRANSACTION EXPLORER
# =========================================================

st.header("Transaction Explorer")

search_text = st.text_input(
    "Search merchant name",
    placeholder="Enter a merchant name...",
)

explorer_df = filtered_df.copy()

if search_text.strip():
    explorer_df = explorer_df[
        explorer_df["Merchant Name"]
        .astype(str)
        .str.contains(
            search_text.strip(),
            case=False,
            na=False,
            regex=False,
        )
    ]

sort_option = st.selectbox(
    "Sort transactions by",
    options=[
        "Newest first",
        "Oldest first",
        "Highest amount first",
        "Lowest amount first",
    ],
)

if sort_option == "Newest first":
    explorer_df = explorer_df.sort_values(
        "Date",
        ascending=False,
    )
elif sort_option == "Oldest first":
    explorer_df = explorer_df.sort_values(
        "Date",
        ascending=True,
    )
elif sort_option == "Highest amount first":
    explorer_df = explorer_df.sort_values(
        "Transaction Amount",
        ascending=False,
    )
else:
    explorer_df = explorer_df.sort_values(
        "Transaction Amount",
        ascending=True,
    )

st.write(
    f"Showing {len(explorer_df):,} matching transactions."
)

st.dataframe(
    explorer_df.drop(columns=["Month"], errors="ignore"),
    use_container_width=True,
    hide_index=True,
    height=400,
)

csv_data = explorer_df.drop(
    columns=["Month"],
    errors="ignore",
).to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered transactions as CSV",
    data=csv_data,
    file_name="filtered_transactions.csv",
    mime="text/csv",
)


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

st.header("Model Evaluation")

if METRICS_PATH.exists():
    try:
        metrics_df = pd.read_csv(METRICS_PATH)

        if not metrics_df.empty:
            metrics = metrics_df.iloc[0]

            if {"Accuracy", "Macro-F1"}.issubset(metrics_df.columns):
                accuracy = float(metrics["Accuracy"])
                macro_f1 = float(metrics["Macro-F1"])

                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.metric(
                        "Test Accuracy",
                        f"{accuracy:.2%}",
                    )

                with metric_col2:
                    st.metric(
                        "Test Macro-F1",
                        f"{macro_f1:.3f}",
                    )

                st.caption(
                    "Metrics are from the saved held-out test evaluation."
                )
            else:
                st.warning(
                    "The metrics file must contain Accuracy and Macro-F1 columns."
                )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Unable to load model metrics: {error}")
else:
    st.warning(
        "Model metrics file not found. "
        "Check reports/model_metrics.csv."
    )

if CONFUSION_PATH.exists():
    try:
        confusion_df = pd.read_csv(
            CONFUSION_PATH,
            index_col=0,
        )

        st.subheader("Confusion Matrix")
        st.caption(
            "Rows and columns represent the actual and predicted categories, "
            "respectively, if saved in that format."
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Unable to load the confusion matrix: {error}")
else:
    st.info(
        "Confusion matrix file not found. "
        "Check reports/confusion_matrix.csv."
    )


# =========================================================
# 10. MACHINE-LEARNING CATEGORY PREDICTION
# =========================================================

st.header("Predict Transaction Category")

st.write(
    "Enter a transaction amount to estimate its likely category "
    "using the trained Logistic Regression model."
)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model was not found. "
        "Run src/train_model.py to generate it."
    )
else:
    try:
        model = load_model(str(MODEL_PATH))

        transaction_amount = st.number_input(
            "Transaction amount",
            min_value=0.01,
            value=500.00,
            step=10.00,
            help="Enter a positive transaction amount.",
        )

        if st.button("Predict Category", type="primary"):
            input_data = pd.DataFrame(
                {"Transaction Amount": [transaction_amount]}
            )

            prediction = model.predict(input_data)[0]

            st.success(f"Predicted category: {prediction}")

            st.info(
                "This is a model estimate, not a guaranteed classification. "
                "The model uses transaction amount alone and achieved "
                "approximately 48.1% accuracy on its held-out test set."
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_

                probability_table = pd.DataFrame(
                    {
                        "Category": classes,
                        "Model Probability": probabilities,
                    }
                ).sort_values(
                    "Model Probability",
                    ascending=False,
                )

                st.subheader("Model Scores by Category")

                st.dataframe(
                    probability_table.style.format(
                        {"Model Probability": "{:.1%}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    except (ValueError, OSError, AttributeError) as error:
        st.error(f"Unable to make a prediction: {error}")


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Expense Analyzer | Exploratory Data Analysis "
    "and Transaction Category Insights"
)