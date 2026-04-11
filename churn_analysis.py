# %% [markdown]
# # Customer Churn Analysis for Retention Insights
#
# **Goodstart Early Learning, Take-Home Challenge**
#
# This notebook addresses both parts of the take-home challenge:
# - **Exercise 1**: Exploratory analysis and insight generation to better understand customer retention.
# - **Exercise 2**: Predictive modelling to estimate churn risk and support proactive intervention.
#
# The goal is not only to build a useful predictive model, but also to surface clear, practical insights
# that can help the Head of Family Experience and the operations team make more data-driven decisions.

# %% [markdown]
# ## 1. Business Objective and Challenge Framing
#
# Goodstart's case study describes a business need to better understand why customers leave, move beyond
# anecdotal explanations, and use historical data to improve retention.
#
# This notebook is structured around two linked questions:
#
# 1. **What patterns in the data help explain customer churn and retention?**
# 2. **Can we build a predictive model that identifies likely churners and highlights useful areas for intervention?**

# %%
import os
import warnings
from typing import Any, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
np.random.seed(42)

PALETTE = {"no_churn": "#2D6A4F", "churn": "#D62828"}
COLORS = [PALETTE["no_churn"], PALETTE["churn"]]
ACCENTS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
BG = "#F8F9FA"
GRID_CLR = "#DEE2E6"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID_CLR,
        "grid.color": GRID_CLR,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
    }
)


# =============================================================================
# Lightweight notebook helpers only
# =============================================================================

def ensure_output_dirs() -> None:
    os.makedirs("images", exist_ok=True)


def section_title(title: str) -> None:
    line = "=" * 88
    print(f"\n{line}\n{title.upper()}\n{line}")


def subsection_title(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def metric_block(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:.4f}")
        else:
            print(f"{key:<28}: {value}")


def display_shape_summary(data: pd.DataFrame, name: str = "Dataset") -> None:
    print(f"{name}: {data.shape[0]:,} rows, {data.shape[1]:,} columns")


def display_missing_summary(data: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    summary = (
        data.isnull()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda x: x["missing_count"] / len(data))
        .sort_values(["missing_count", "missing_pct"], ascending=False)
        .head(top_n)
    )
    print("\nTop Missingness Summary")
    print(summary)
    return summary


def display_target_summary(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    summary = (
        data[target_col]
        .value_counts(dropna=False)
        .rename_axis(target_col)
        .reset_index(name="count")
    )
    summary["pct"] = summary["count"] / summary["count"].sum()
    print(f"\nTarget Summary: {target_col}")
    print(summary)
    return summary


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


ensure_output_dirs()

# %% [markdown]
# ## 2. Dataset Overview and Assumptions
#
# Key assumptions used in this notebook:
#
# - The supplied target column, `churn_risk_score`, is treated as the modelling target.
# - The working dataset contains a binary target, so this is treated as a **binary classification** problem.
# - This is a historical static dataset, not a true event-level churn timeline.
# - Some variables, especially complaint and feedback fields, may be close to the outcome timing and should be interpreted carefully.

# %%
section_title("Load Data")

data_path = "/mnt/d/GreenJourney/DataScienceProjects/churn-prediction-gs/data/churn.csv"
df = pd.read_csv(data_path)

display_shape_summary(df, "Raw dataset")
print("\nPreview:")
display(df.head())

# %% [markdown]
# ## 3. Exercise 1, Exploratory Analysis and Business Insights

# %% [markdown]
# ### 3.1 Data Quality Review
#
# The first step is to clean placeholder values, sentinel values, and invalid numeric entries so that
# the downstream analysis and modelling are based on a more defensible dataset.

# %%
section_title("Exercise 1 - Data Quality Review")

df_clean = df.copy()
df_clean.columns = [c.strip() for c in df_clean.columns]

placeholder_values = ["?", "Unknown", "unknown", "NA", "N/A", ""]
object_cols = df_clean.select_dtypes(include="object").columns
df_clean[object_cols] = df_clean[object_cols].replace(placeholder_values, np.nan)

if "referral_id" in df_clean.columns:
    df_clean["referral_id"] = df_clean["referral_id"].replace(r"^x+$", np.nan, regex=True)

numeric_cols = [
    "age",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
]

if "avg_frequency_login_days" in df_clean.columns:
    df_clean["avg_frequency_login_days"] = df_clean["avg_frequency_login_days"].replace("Error", np.nan)

for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

df_clean["days_since_last_login"] = df_clean["days_since_last_login"].replace(-999, np.nan)

invalid_specs = {
    "avg_time_spent": "avg_time_spent_invalid",
    "avg_frequency_login_days": "avg_frequency_login_days_invalid",
    "points_in_wallet": "points_in_wallet_invalid",
}

for source_col, flag_col in invalid_specs.items():
    df_clean[flag_col] = df_clean[source_col].lt(0).astype(int)
    df_clean.loc[df_clean[source_col] < 0, source_col] = np.nan

df_clean["complaint_status"] = df_clean["complaint_status"].fillna("Missing")
df_clean["feedback"] = df_clean["feedback"].fillna("Missing")

display_shape_summary(df_clean, "Cleaned dataset")
missing_summary = display_missing_summary(df_clean, top_n=15)

print("\nNumeric Feature Summary After Cleaning:")
display(df_clean[numeric_cols].describe().T)

print("\nData types after cleaning:")
display(df_clean.dtypes.to_frame("dtype"))

# %% [markdown]
# ### Optional automated profiling export

# %%
try:
    from ydata_profiling import ProfileReport

    section_title("Automated EDA Profiling")

    profile = ProfileReport(
        df_clean,
        title="Customer Churn, Cleaned Data Profiling Report",
        explorative=True,
        minimal=False,
    )
    profile.to_file("churn_eda_report.html")
    print("EDA report saved to churn_eda_report.html")

except Exception as e:
    print(f"ydata-profiling could not run: {e}")

# %% [markdown]
# ### 3.2 Data Structure and Initial Checks

# %%
id_columns = ["Unnamed: 0", "security_no", "referral_id"]
date_columns = ["joining_date", "last_visit_time"]
target = "churn_risk_score"

base_categorical_columns = [
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "used_special_discount",
    "offer_application_preference",
    "feedback",
    "past_complaint",
    "complaint_status",
]

metric_block(
    "Column Categorization",
    {
        "ID columns": len(id_columns),
        "Date columns": len(date_columns),
        "Categorical columns": len(base_categorical_columns),
        "Numerical columns": len(numeric_cols),
        "Target variable": target,
    },
)

print(f"\nDuplicate rows after cleaning: {df_clean.duplicated().sum()}")

target_summary = display_target_summary(df_clean, target)
print(f"\nOverall churn rate: {df_clean[target].mean():.2%}")
print("\nThe supplied dataset contains a binary target column, so the task is treated as binary classification.")

# %% [markdown]
# ### 3.3 Categorical Audit

# %%
categorical_audit_cols = [
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "used_special_discount",
    "offer_application_preference",
    "past_complaint",
    "complaint_status",
    "feedback",
]

for col in categorical_audit_cols:
    print(f"\nColumn: {col}")
    print("-" * (8 + len(col)))
    print(f"Unique values: {df_clean[col].nunique(dropna=False)}")
    display(df_clean[col].value_counts(dropna=False).to_frame("count"))

# %% [markdown]
# ### 3.4 Feature Engineering for Exploration and Interpretation
#
# Business-specific transformations are defined here, close to the point of use, while keeping
# repeated binning logic concise and consistent.

# %%
section_title("Feature Engineering")


def add_missing_label_from_source(
    category_series: pd.Series,
    source_series: pd.Series,
    label: str,
) -> pd.Series:
    output = category_series.astype("object")
    output.loc[source_series.isna()] = label
    return output


def build_binned_feature(
    data: pd.DataFrame,
    source_col: str,
    bins,
    labels,
    missing_label: str,
    right: bool = True,
) -> pd.Series:
    grouped = pd.cut(data[source_col], bins=bins, labels=labels, right=right)
    return add_missing_label_from_source(grouped, data[source_col], missing_label)

df_clean["joining_date"] = pd.to_datetime(df_clean["joining_date"], errors="coerce")
df_clean["last_visit_hour"] = pd.to_datetime(
    df_clean["last_visit_time"], format="%H:%M:%S", errors="coerce"
).dt.hour

reference_date = df_clean["joining_date"].max()
df_clean["relative_tenure_days"] = (reference_date - df_clean["joining_date"]).dt.days

df_clean["engagement_score"] = df_clean["avg_time_spent"] * df_clean["avg_frequency_login_days"]
df_clean["value_per_login"] = df_clean["avg_transaction_value"] / (df_clean["avg_frequency_login_days"] + 1)

# Business grouping dictionaries
feedback_map = {
    "Poor Website": "Negative",
    "Poor Product Quality": "Negative",
    "Poor Customer Service": "Negative",
    "Too many ads": "Negative",
    "Poor User Experience": "Negative",
    "No reason specified": "Negative",
    "Quality Customer Care": "Positive",
    "User Friendly Website": "Positive",
    "Products always in Stock": "Positive",
    "Missing": "Missing",
}

complaint_map = {
    "Not Applicable": "No Complaint",
    "Solved": "Resolved",
    "Solved in Follow-up": "Resolved",
    "Unsolved": "Unresolved_or_Unknown",
    "No Information Available": "Unresolved_or_Unknown",
    "Missing": "Missing",
}

df_clean["feedback_sentiment"] = df_clean["feedback"].map(feedback_map).fillna("Other")
df_clean["complaint_status_group"] = df_clean["complaint_status"].map(complaint_map).fillna("Other")

df_clean["had_complaint"] = df_clean["past_complaint"].eq("Yes").astype(int)
df_clean["complaint_unresolved"] = df_clean["complaint_status_group"].eq("Unresolved_or_Unknown").astype(int)

df_clean["age_group"] = build_binned_feature(
    df_clean,
    source_col="age",
    bins=[0, 25, 45, 65, np.inf],
    labels=["Young", "Adult", "Mature", "Senior"],
    missing_label="Missing",
    right=False,
)

df_clean["login_recency_group"] = build_binned_feature(
    df_clean,
    source_col="days_since_last_login",
    bins=[-np.inf, 7, 14, 21, np.inf],
    labels=["Recent", "Moderately Inactive", "Inactive", "Highly Inactive"],
    missing_label="Missing",
)

df_clean["time_spent_group"] = build_binned_feature(
    df_clean,
    source_col="avg_time_spent",
    bins=[-np.inf, 60, 180, np.inf],
    labels=["Low", "Medium", "High"],
    missing_label="Invalid_or_Missing",
)

df_clean["transaction_value_group"] = build_binned_feature(
    df_clean,
    source_col="avg_transaction_value",
    bins=[-np.inf, 10000, 30000, 60000, np.inf],
    labels=["Low Value", "Medium Value", "High Value", "Very High Value"],
    missing_label="Missing",
)

df_clean["login_frequency_group"] = build_binned_feature(
    df_clean,
    source_col="avg_frequency_login_days",
    bins=[-np.inf, 10, 20, 35, np.inf],
    labels=["Low Frequency", "Moderate Frequency", "High Frequency", "Very High Frequency"],
    missing_label="Invalid_or_Missing",
)

df_clean["wallet_points_group"] = build_binned_feature(
    df_clean,
    source_col="points_in_wallet",
    bins=[-np.inf, 500, 800, 1200, np.inf],
    labels=["Low Points", "Medium Points", "High Points", "Very High Points"],
    missing_label="Invalid_or_Missing",
)

visit_period_conditions = [
    df_clean["last_visit_hour"].between(5, 11, inclusive="both"),
    df_clean["last_visit_hour"].between(12, 16, inclusive="both"),
    df_clean["last_visit_hour"].between(17, 20, inclusive="both"),
]
visit_period_choices = ["Morning", "Afternoon", "Evening"]
df_clean["visit_period"] = np.select(visit_period_conditions, visit_period_choices, default="Night")
df_clean.loc[df_clean["last_visit_hour"].isna(), "visit_period"] = "Missing"

df_clean["tenure_group"] = pd.cut(
    df_clean["relative_tenure_days"],
    bins=[-1, 180, 365, 730, np.inf],
    labels=["New", "Established", "Loyal", "Long-Term"],
)
df_clean["relative_tenure_bin"] = pd.cut(df_clean["relative_tenure_days"], bins=5)
df_clean["joining_month"] = df_clean["joining_date"].dt.to_period("M")

engineered_feature_cols = [
    "feedback_sentiment",
    "complaint_status_group",
    "age_group",
    "login_recency_group",
    "time_spent_group",
    "transaction_value_group",
    "login_frequency_group",
    "wallet_points_group",
    "visit_period",
    "tenure_group",
    "avg_time_spent_invalid",
    "avg_frequency_login_days_invalid",
    "points_in_wallet_invalid",
    "had_complaint",
    "complaint_unresolved",
    "relative_tenure_days",
    "engagement_score",
    "value_per_login",
]

print("Created grouped and business-friendly features:")
for col in engineered_feature_cols:
    print(f"  • {col}")

display(df_clean[engineered_feature_cols].head())

# %% [markdown]
# ### 3.5 Exploratory Analysis

# %%
section_title("Exploratory Analysis")

counts = df_clean[target].value_counts().sort_index()
labels = ["No Churn", "Churn"]
churn_rate = df_clean[target].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Customer Churn Overview", fontsize=15, fontweight="bold")

bars = axes[0].bar(labels, counts.values, color=COLORS, width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 150,
        f"{val:,}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
axes[0].set_ylim(0, counts.max() * 1.18)
style_ax(axes[0], "Class Distribution", "", "Number of Customers")

wedges, texts, autotexts = axes[1].pie(
    counts.values,
    labels=labels,
    colors=COLORS,
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.75,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2.5},
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight("bold")
    at.set_color("white")

axes[1].set_title(f"Churn Rate: {churn_rate:.1%}", pad=10, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Monthly cohort trend
trend_df = (
    df_clean.assign(join_year_month=df_clean["joining_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("join_year_month")
    .agg(churn_rate=(target, "mean"), customer_count=(target, "size"))
    .reset_index()
    .sort_values("join_year_month")
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    trend_df["join_year_month"],
    trend_df["churn_rate"],
    marker="o",
    linewidth=2.2,
    color=ACCENTS[4],
    markersize=5,
)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
style_ax(ax, "Churn Rate by Customer Cohort, Joining Month", "Joining Month", "Churn Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/cohort_trend.png", dpi=200, bbox_inches="tight")
plt.show()

tenure_churn = df_clean.groupby("relative_tenure_bin")[target].mean()

fig, ax = plt.subplots(figsize=(8, 5))
tenure_churn.plot(kind="bar", ax=ax, color=ACCENTS[1], edgecolor="white")
for bar, val in zip(ax.patches, tenure_churn):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
style_ax(ax, "Churn Rate by Relative Tenure", "Relative Tenure Bucket", "Churn Rate")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

cat_features = [
    ("membership_category", "Membership Category"),
    ("feedback_sentiment", "Feedback Sentiment"),
    ("complaint_status_group", "Complaint Status Group"),
    ("login_recency_group", "Login Recency Group"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Churn Patterns Across Key Customer Segments", fontsize=15, fontweight="bold")

for ax, (col, title) in zip(axes.flat, cat_features):
    churn_rate_by_cat = df_clean.groupby(col)[target].mean().sort_values()
    counts_by_cat = df_clean[col].value_counts(dropna=False)

    churn_rate_by_cat.plot(kind="barh", ax=ax, color=ACCENTS[1], edgecolor="white")
    for i, (cat, val) in enumerate(churn_rate_by_cat.items()):
        n = counts_by_cat.get(cat, 0)
        ax.text(val + 0.005, i, f"{val:.2f} (n={n:,})", va="center", fontsize=8.5)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, churn_rate_by_cat.max() * 1.35 if churn_rate_by_cat.max() > 0 else 1)
    style_ax(ax, title, "Churn Rate", "")

plt.tight_layout()
plt.show()

engineered_numerical = [
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "relative_tenure_days",
    "engagement_score",
    "points_in_wallet",
]
plot_df = df_clean[engineered_numerical + [target]].copy()
plot_df[target] = plot_df[target].map({0: "No Churn", 1: "Churn"})

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribution of Core Numeric Drivers by Churn Status", fontsize=15, fontweight="bold")

for ax, col in zip(axes.flat, engineered_numerical):
    sns.violinplot(
        data=plot_df,
        x=target,
        y=col,
        palette=COLORS,
        order=["No Churn", "Churn"],
        inner=None,
        linewidth=0.8,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x=target,
        y=col,
        order=["No Churn", "Churn"],
        width=0.14,
        color="white",
        linewidth=1.2,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        ax=ax,
    )
    style_ax(ax, col.replace("_", " ").title(), "", "")

plt.tight_layout()
plt.show()

grouped_eda_cols = [
    "feedback_sentiment",
    "complaint_status_group",
    "age_group",
    "login_recency_group",
    "time_spent_group",
    "transaction_value_group",
    "login_frequency_group",
    "wallet_points_group",
    "visit_period",
    "tenure_group",
]

for col in grouped_eda_cols:
    summary = (
        df_clean.groupby(col, dropna=False)[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )
    print(f"\nTarget summary by {col}")
    display(summary)

# %% [markdown]
# ### 3.6 Key Insights for the Business
#
# - churn is not extremely imbalanced, which supports standard classification approaches
# - grouped behavioural features such as login recency, time spent, and transaction value provide
#   more business-friendly insight than raw values alone
# - complaint and feedback related variables appear important, though they may also be close to the
#   target timing and should be interpreted carefully
# - the data quality issues are material enough that cleaning choices need to be clearly documented

# %% [markdown]
# ## 4. Exercise 2, Predictive Modelling and Advanced Insights

# %% [markdown]
# ### 4.1 Modelling Objective
#
# The modelling objective is to predict the binary churn target provided in the dataset and produce
# insights that can help the Family Experience team prioritise intervention.

# %% [markdown]
# ### 4.2 Modelling Assumptions and Design Choices
#
# Main modelling decisions:
#
# - **Problem type**: binary classification, using the supplied target
# - **Features**: raw numeric variables retained for predictive strength, grouped variables added for interpretability
# - **Validation**: stratified train-test split and stratified cross-validation on the training set
# - **Metrics**: F1, confusion matrix, precision, recall, and ROC-AUC
# - **Explainability**: feature importance where available, plus permutation importance for model-agnostic interpretation

# %%
section_title("Exercise 2 - Prepare Data for Modelling")

drop_columns = id_columns + date_columns + ["joining_month", "relative_tenure_bin"]

df_model = df_clean.drop(columns=[c for c in drop_columns if c in df_clean.columns])

raw_numeric_features = [
    "age",
    "days_since_last_login",
    "avg_time_spent",
    "avg_transaction_value",
    "avg_frequency_login_days",
    "points_in_wallet",
    "relative_tenure_days",
    "engagement_score",
    "value_per_login",
    "last_visit_hour",
]

flag_features = [
    "avg_time_spent_invalid",
    "avg_frequency_login_days_invalid",
    "points_in_wallet_invalid",
    "had_complaint",
    "complaint_unresolved",
]

grouped_categorical_features = [
    "gender",
    "region_category",
    "membership_category",
    "joined_through_referral",
    "preferred_offer_types",
    "medium_of_operation",
    "internet_option",
    "used_special_discount",
    "offer_application_preference",
    "feedback_sentiment",
    "complaint_status_group",
    "age_group",
    "login_recency_group",
    "time_spent_group",
    "transaction_value_group",
    "login_frequency_group",
    "wallet_points_group",
    "visit_period",
    "tenure_group",
]

X = df_model.drop(columns=[target])
y = df_model[target]

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"\nTarget distribution:\n{y.value_counts()}")

# %% [markdown]
# ### 4.3 Train-Test Split

# %%
section_title("Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTrain churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")

# %% [markdown]
# ### 4.4 Preprocessing Pipeline

# %%
section_title("Preprocessing Pipelines")


def evaluate_candidate_on_test(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Test F1": f1_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred),
        "Test Recall": recall_score(y_test, y_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def evaluate_model(
    pipeline: Pipeline,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    model_name: str,
) -> Dict[str, float]:
    y_train_pred = pipeline.predict(X_tr)
    y_test_pred = pipeline.predict(X_te)
    y_test_proba = pipeline.predict_proba(X_te)[:, 1]

    print(f"\n{'=' * 60}")
    print(f"{model_name} - Final Evaluation")
    print(f"{'=' * 60}")

    print("\nTrain Metrics:")
    print(f"  F1:        {f1_score(y_tr, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_tr, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_tr, y_train_pred):.4f}")

    print("\nTest Metrics:")
    print(f"  F1:        {f1_score(y_te, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_te, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_te, y_test_proba):.4f}")

    print("\nClassification Report, Test:")
    print(classification_report(y_te, y_test_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_te, y_test_pred)
    fpr, tpr, _ = roc_curve(y_te, y_test_proba)
    auc_val = roc_auc_score(y_te, y_test_proba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} - Final Evaluation on Held-Out Test Set", fontsize=14, fontweight="bold")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array(
        [
            [f"{count:,}\n({pct:.1f}%)" for count, pct in zip(row_counts, row_pcts)]
            for row_counts, row_pcts in zip(cm, cm_pct)
        ]
    )

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 11},
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=axes[0],
        cbar_kws={"shrink": 0.8},
    )
    style_ax(axes[0], "Confusion Matrix", "Predicted", "Actual")

    axes[1].plot(fpr, tpr, color=ACCENTS[4], lw=2.5, label=f"AUC = {auc_val:.4f}")
    axes[1].fill_between(fpr, tpr, alpha=0.08, color=ACCENTS[4])
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    axes[1].set_xlim([-0.01, 1.01])
    axes[1].set_ylim([-0.01, 1.05])
    axes[1].legend(loc="lower right", frameon=True)
    style_ax(axes[1], "ROC Curve", "False Positive Rate", "True Positive Rate")

    plt.tight_layout()
    plt.savefig("images/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.show()

    return {
        "f1": f1_score(y_te, y_test_pred),
        "precision": precision_score(y_te, y_test_pred),
        "recall": recall_score(y_te, y_test_pred),
        "roc_auc": roc_auc_score(y_te, y_test_proba),
    }


def plot_importance(df_imp: pd.DataFrame, value_col: str, title: str, xlabel: str) -> None:
    top = df_imp.head(15).copy()
    top["label"] = top["feature"].astype(str).str.replace(r"^[a-z_]+_", "", regex=True)

    bar_colors = [ACCENTS[1] if value >= 0 else PALETTE["churn"] for value in top[value_col]]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top["label"], top[value_col], color=bar_colors, edgecolor="white", linewidth=0.8)

    max_val = max(top[value_col].abs().max(), 1e-9)
    for bar, value in zip(bars, top[value_col]):
        x_pos = value + max_val * 0.015 if value >= 0 else value - max_val * 0.015
        ha = "left" if value >= 0 else "right"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax.invert_yaxis()
    ax.axvline(0, color="#888", linewidth=0.8)
    style_ax(ax, title, xlabel, "")
    plt.tight_layout()
    plt.show()

raw_numeric_features = [c for c in raw_numeric_features if c in X.columns]
flag_features = [c for c in flag_features if c in X.columns]
nominal_cats = [c for c in grouped_categorical_features if c in X.columns]

num_lr = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

num_tree = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

cat_pipe_sparse = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

try:
    dense_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    dense_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_pipe_dense = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", dense_encoder),
    ]
)

lr_preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_lr, raw_numeric_features),
        ("cat", cat_pipe_sparse, nominal_cats),
        ("flag", "passthrough", flag_features),
    ]
)

tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_tree, raw_numeric_features),
        ("cat", cat_pipe_sparse, nominal_cats),
        ("flag", "passthrough", flag_features),
    ]
)

gb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_tree, raw_numeric_features),
        ("cat", cat_pipe_dense, nominal_cats),
        ("flag", "passthrough", flag_features),
    ]
)

lr_pipeline = Pipeline(
    [
        ("preprocessor", lr_preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ]
)

rf_pipeline = Pipeline(
    [
        ("preprocessor", tree_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)

gb_pipeline = Pipeline(
    [
        ("preprocessor", gb_preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
        ),
    ]
)

xgb_pipeline = Pipeline(
    [
        ("preprocessor", tree_preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        ),
    ]
)

print("Pipelines defined successfully.")

# %% [markdown]
# ### 4.5 Baseline and Model Comparison

# %%
section_title("Baseline and Model Comparison")

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print(f"Baseline, majority class, F1: {f1_score(y_test, y_dummy):.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline,
    "Gradient Boosting": gb_pipeline,
    "XGBoost": xgb_pipeline,
}

cv_results: Dict[str, np.ndarray] = {}
print("\nRunning 5-fold stratified cross-validation on training set, scoring by F1.\n")

for name, pipeline in candidates.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name}: mean F1 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

cv_summary = pd.DataFrame(
    {
        name: {"cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()}
        for name, scores in cv_results.items()
    }
).T

print("\nCross-validation summary:")
display(cv_summary)

fig, ax = plt.subplots(figsize=(9, 5))
means = cv_summary["cv_f1_mean"]
stds = cv_summary["cv_f1_std"]
best = means.idxmax()

bar_colors = [ACCENTS[4] if name == best else ACCENTS[1] for name in means.index]
bars = ax.barh(
    means.index,
    means,
    xerr=stds,
    color=bar_colors,
    capsize=5,
    edgecolor="white",
    linewidth=1.1,
    height=0.5,
    error_kw={"elinewidth": 1.5, "ecolor": "#555"},
)

for bar, val, std in zip(bars, means, stds):
    ax.text(
        val + std + 0.003,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f} ± {std:.4f}",
        va="center",
        fontsize=10,
    )

ax.set_xlim(max(0, means.min() - 0.03), min(1.02, means.max() + stds.max() + 0.03))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.axvline(means.max(), color=ACCENTS[4], linestyle="--", linewidth=1, alpha=0.5)
ax.invert_yaxis()
style_ax(ax, "5-Fold Stratified CV, F1 Score", "Mean F1 Score", "")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.6 Final Comparison Across Tested Models

# This section complements cross-validation by showing held-out test results for
# each model considered in the comparison.

# %%
model_test_results = []
for name, pipeline in candidates.items():
    result = evaluate_candidate_on_test(name, pipeline, X_train, X_test, y_train, y_test)
    result["CV F1 Mean"] = cv_summary.loc[name, "cv_f1_mean"]
    result["CV F1 Std"] = cv_summary.loc[name, "cv_f1_std"]
    model_test_results.append(result)

comparison_df = pd.DataFrame(model_test_results)
comparison_df = comparison_df[
    ["Model", "CV F1 Mean", "CV F1 Std", "Test F1", "Test Precision", "Test Recall", "Test ROC-AUC"]
].sort_values("CV F1 Mean", ascending=False)

print("\nModel comparison summary:")
display(
    comparison_df.style.format(
        {
            "CV F1 Mean": "{:.4f}",
            "CV F1 Std": "{:.4f}",
            "Test F1": "{:.4f}",
            "Test Precision": "{:.4f}",
            "Test Recall": "{:.4f}",
            "Test ROC-AUC": "{:.4f}",
        }
    )
)

fig, ax = plt.subplots(figsize=(10, 5))
plot_df = comparison_df.sort_values("CV F1 Mean", ascending=True)

bars = ax.barh(
    plot_df["Model"],
    plot_df["CV F1 Mean"],
    xerr=plot_df["CV F1 Std"],
    color=ACCENTS[1],
    edgecolor="white",
    linewidth=1.0,
    capsize=4,
)

for bar, val in zip(bars, plot_df["CV F1 Mean"]):
    ax.text(
        val + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center",
        fontsize=9,
    )

style_ax(ax, "Model Comparison, Cross-Validated F1", "Mean CV F1", "")
ax.set_xlim(max(0, plot_df["CV F1 Mean"].min() - 0.03), min(1.00, plot_df["CV F1 Mean"].max() + plot_df["CV F1 Std"].max() + 0.03))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.7 Selected Model and Final Held-Out Evaluation

# %%
section_title("Selected Model and Final Held-Out Evaluation")

best_model_name = cv_summary["cv_f1_mean"].idxmax()
best_pipeline = candidates[best_model_name]

print(f"Selected model: {best_model_name}")
print(f"CV F1 mean: {cv_summary.loc[best_model_name, 'cv_f1_mean']:.4f}")
print("Fitting selected model on full training set...")

best_pipeline.fit(X_train, y_train)

final_metrics = evaluate_model(best_pipeline, X_train, X_test, y_train, y_test, best_model_name)

y_probs = best_pipeline.predict_proba(X_test)[:, 1]
print("\nThreshold sensitivity:")
for threshold in [0.4, 0.5, 0.6]:
    y_custom = (y_probs > threshold).astype(int)
    print(
        f"  threshold={threshold:.1f}  "
        f"F1={f1_score(y_test, y_custom):.4f}  "
        f"Precision={precision_score(y_test, y_custom):.4f}  "
        f"Recall={recall_score(y_test, y_custom):.4f}"
    )

# %% [markdown]
# ### 4.8 Explainability and Advanced Insights

# %%
section_title("Explainability and Advanced Insights")

try:
    ohe_feature_names = (
        best_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(nominal_cats)
        .tolist()
    )
    all_feature_names = raw_numeric_features + ohe_feature_names + flag_features
except Exception:
    all_feature_names = list(X.columns)

inner_model = best_pipeline.named_steps["model"]

if hasattr(inner_model, "feature_importances_"):
    feature_importance = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": inner_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print(f"\nTop 15 Feature Importances, {best_model_name}:")
    print(feature_importance.head(15).to_string(index=False))
    plot_importance(feature_importance, "importance", f"{best_model_name} - Top Feature Importances", "Importance")

elif hasattr(inner_model, "coef_"):
    feature_importance = pd.DataFrame(
        {
            "feature": all_feature_names,
            "coefficient": inner_model.coef_[0],
        }
    ).sort_values("coefficient", key=abs, ascending=False)

    print(f"\nTop 15 Feature Coefficients, {best_model_name}:")
    print(feature_importance.head(15).to_string(index=False))
    plot_importance(feature_importance, "coefficient", f"{best_model_name} - Top Coefficients", "Coefficient")

print("\nCalculating permutation importance on test set...")
X_test_transformed = best_pipeline.named_steps["preprocessor"].transform(X_test)

perm_imp = permutation_importance(
    inner_model,
    X_test_transformed,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

perm_importance_df = pd.DataFrame(
    {
        "feature": all_feature_names,
        "importance": perm_imp.importances_mean,
    }
).sort_values("importance", ascending=False)

print("\nTop 15 Permutation Importances:")
print(perm_importance_df.head(15).to_string(index=False))
plot_importance(perm_importance_df, "importance", "Top Features, Permutation Importance", "Mean Decrease in F1")

# %% [markdown]
# ## 5. Business Recommendations and Production Considerations

# %%
section_title("Business Recommendations and Production Considerations")

print("\n1. Key business insights")
print("-" * 80)
top_features = perm_importance_df.head(10)["feature"].tolist()
print("The strongest predictive signals include:")
for i, feature in enumerate(top_features[:5], 1):
    print(f"   {i}. {feature}")

print("\n2. Recommended action areas")
print("-" * 80)
print("   • Review negative feedback and unresolved complaint patterns closely")
print("   • Prioritise inactive and highly inactive customers for re-engagement")
print("   • Investigate whether lower value or lower wallet-point customers need targeted retention treatment")
print("   • Review membership segments with higher predicted churn risk")

print("\n3. Field testing approach")
print("-" * 80)
print("   • Run an A/B test, control versus treatment, using model-driven interventions")
print("   • Compare retention uplift and operational cost of outreach")
print("   • Validate score quality on future data before broad rollout")

print("\n4. Productionisation considerations")
print("-" * 80)
print("   • Confirm all model features are available at scoring time in real workflows")
print("   • Automate refresh of input data and scoring outputs")
print("   • Integrate outputs into CRM or customer operations processes")
print("   • Monitor drift, missingness, and score performance over time")

print("\n5. Expected business usefulness")
print("-" * 80)
print(f"   • Recall:    {final_metrics['recall']:.1%} of churners identified")
print(f"   • Precision: {final_metrics['precision']:.1%} of flagged customers are true churners")
print("   • These outputs can support prioritised outreach rather than relying only on anecdotal judgement")

# %% [markdown]
# ## 6. Limitations and Important Caveats

# %%
print("- The supplied target is used as given, and treated as binary classification.")
print("- The working dataset supports a 0/1 target.")
print("- No true churn timestamp is available, so this is not a full event-time churn model.")
print("- relative_tenure_days is relative to the latest joining date in the dataset, not true production tenure at scoring time.")
print("- Some fields, especially complaint and feedback related variables, may be close to the outcome timing.")
print("- Feature importance is predictive, not causal.")
print("- Out-of-time validation remains necessary before production deployment.")

# %% [markdown]
# ## 7. Final Summary

# %%
section_title("Final Summary")

metric_block(
    "Model Performance Summary",
    {
        "Selected model": best_model_name,
        "CV F1, mean": cv_summary.loc[best_model_name, "cv_f1_mean"],
        "Test F1": final_metrics["f1"],
        "Test Recall": final_metrics["recall"],
        "Test ROC-AUC": final_metrics["roc_auc"],
    },
)

print(
    "\nThe final workflow combines exploratory analysis, feature engineering, predictive modelling, "
    "model comparison, and explainability. It is intended as a practical and interpretable starting point "
    "for proactive retention analysis, while still requiring future-data validation and production checks."
)
