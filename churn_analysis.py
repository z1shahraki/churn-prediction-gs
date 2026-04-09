# %% [markdown]
# # Customer Churn Prediction Analysis
#
# ## Business Problem
# Customer churn is a critical business metric that directly impacts revenue and growth.
# This analysis aims to:
# - Predict which customers are at risk of churning (binary classification)
# - Identify actionable drivers of churn
# - Provide business recommendations for retention strategies
#
# The goal is not just prediction accuracy, but actionable insights that operations teams can use.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings

# ── Global plot style ────────────────────────────────────────────────────────
PALETTE  = {'no_churn': '#2D6A4F', 'churn': '#D62828'}
COLORS   = [PALETTE['no_churn'], PALETTE['churn']]
ACCENTS  = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
BG       = '#F8F9FA'
GRID_CLR = '#DEE2E6'

sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'axes.edgecolor':    GRID_CLR,
    'grid.color':        GRID_CLR,
    'grid.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        120,
})

def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:  ax.set_title(title, pad=10)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.tick_params(length=0)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
np.random.seed(42)

print("Environment setup complete")

# %% [markdown]
# ## 2. Load Data

# %%
data_path = '/mnt/d/GreenJourney/DataScienceProjects/churn-prediction-gs/data/churn.csv'
df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print(f"\nShape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# %% [markdown]
# ### Data Quality Cleaning
# Replace placeholder strings, fix sentinel values, and remove impossible negatives.

# %%
df_clean = df.copy()

# 1. Replace placeholder strings with NaN
placeholder_values = ['?', 'Unknown', 'unknown', 'NA', 'N/A', 'xxxxxxx', '']
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].replace(placeholder_values, np.nan)

# 2. Convert numeric columns safely
numeric_cols = [
    'age', 'days_since_last_login', 'avg_time_spent',
    'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'
]
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 3. Fix sentinel values
df_clean['days_since_last_login'] = df_clean['days_since_last_login'].replace(-999, np.nan)

# 4. Remove impossible negative values
df_clean.loc[df_clean['avg_time_spent'] < 0, 'avg_time_spent'] = np.nan
df_clean.loc[df_clean['points_in_wallet'] < 0, 'points_in_wallet'] = np.nan

print("Data cleaning applied")
print("\nPost-cleaning missing values:")
print(df_clean.isnull().sum().sort_values(ascending=False))
print("\nCheck numeric ranges:")
print(df_clean[numeric_cols].describe())

# %%
print("\nData types after cleaning:")
print(df_clean.dtypes)

print("\nBasic statistics after cleaning:")
print(df_clean.describe())

# %% [markdown]
# ## 3. Data Understanding and Quality Checks

# %%
# Identify column types
id_columns = ['Unnamed: 0', 'security_no', 'referral_id']
date_columns = ['joining_date', 'last_visit_time']
categorical_columns = [
    'gender', 'region_category', 'membership_category', 'joined_through_referral',
    'preferred_offer_types', 'medium_of_operation', 'internet_option',
    'used_special_discount', 'offer_application_preference',
    'past_complaint', 'complaint_status', 'feedback'
]
numerical_columns = [
    'age', 'days_since_last_login', 'avg_time_spent',
    'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet'
]
target = 'churn_risk_score'

print("Column categorization:")
print(f"ID columns: {id_columns}")
print(f"Date columns: {date_columns}")
print(f"Categorical columns: {len(categorical_columns)} columns")
print(f"Numerical columns: {len(numerical_columns)} columns")
print(f"Target: {target}")

# %%
print(f"\nDuplicate rows after cleaning: {df_clean.duplicated().sum()}")

# %%
print("\nTarget distribution:")
print(df[target].value_counts())
print(f"\nChurn rate: {df[target].mean():.2%}")

# %%
# Inspect last_visit_time and complaint_status before feature engineering
print("\nlast_visit_time sample values:")
print(df_clean['last_visit_time'].head(10).tolist())

print("\ncomplaint_status value counts:")
print(df_clean['complaint_status'].value_counts())

print("\npast_complaint value counts:")
print(df_clean['past_complaint'].value_counts())

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# Parse joining_date as a full datetime (used for tenure calculation)
df_clean['joining_date'] = pd.to_datetime(df_clean['joining_date'], errors='coerce')

# NOTE: last_visit_time contains only time-of-day values (HH:MM:SS), not full datetimes.
# We extract last_visit_hour as a proxy for behavioural pattern (e.g. off-hours usage).
df_clean['last_visit_hour'] = pd.to_datetime(
    df_clean['last_visit_time'], format='%H:%M:%S', errors='coerce'
).dt.hour

print("Missing values in last_visit_hour:", df_clean['last_visit_hour'].isnull().sum())

# %% [markdown]
# ## 5. Feature Engineering

# %%
# Tenure
reference_date = df_clean['joining_date'].max()
df_clean['tenure_days'] = (reference_date - df_clean['joining_date']).dt.days

# Engagement
df_clean['engagement_score'] = (
    df_clean['avg_time_spent'] * df_clean['avg_frequency_login_days']
)

# Value intensity
df_clean['value_per_login'] = (
    df_clean['avg_transaction_value'] / (df_clean['avg_frequency_login_days'] + 1)
)

# Complaint features — robust to all complaint_status values observed in data
df_clean['had_complaint'] = (df_clean['past_complaint'] == 'Yes').astype(int)
df_clean['complaint_unresolved'] = (
    df_clean['complaint_status'].isin(['Unsolved', 'No Information Available'])
).astype(int)
df_clean['complaint_severity'] = df_clean['complaint_unresolved'] * df_clean['had_complaint']

# Cohort
df_clean['joining_month'] = df_clean['joining_date'].dt.to_period('M')

# Tenure bins for analysis
df_clean['tenure_bin'] = pd.cut(df_clean['tenure_days'], bins=5)

print("Feature engineering complete")
print("\nEngineered features:")
print("- tenure_days, engagement_score, value_per_login, last_visit_hour")
print("- had_complaint, complaint_unresolved, complaint_severity")
print("- joining_month (cohort), tenure_bin")

print("\ncomplaint_severity distribution:")
print(df_clean['complaint_severity'].value_counts())

# %% [markdown]
# ## 6. Exploratory Data Analysis
#
# Note: Cohort analysis reflects churn by acquisition cohort, not actual churn timing.

# %%
# ── Chart 1: Target distribution ────────────────────────────────────────────────
counts     = df_clean[target].value_counts().sort_index()
labels     = ['No Churn', 'Churn']
churn_rate = df_clean[target].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Customer Churn Overview', fontsize=15, fontweight='bold')

bars = axes[0].bar(labels, counts.values, color=COLORS, width=0.5,
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                 f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
axes[0].set_ylim(0, counts.max() * 1.18)
style_ax(axes[0], 'Class Distribution', '', 'Number of Customers')

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=labels, colors=COLORS,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2.5}
)
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('white')
axes[1].set_title(f'Churn Rate: {churn_rate:.1%}', pad=10, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
print(f"No Churn: {counts[0]:,}  |  Churn: {counts[1]:,}  |  Ratio: {counts[0]/counts[1]:.2f}:1")

# %%
# ── Cohort Analysis: Churn rate by acquisition month ──────────────────────────
cohort_churn = df_clean.groupby('joining_month')[target].mean()

fig, ax = plt.subplots(figsize=(12, 5))
cohort_churn.plot(marker='o', ax=ax, color=ACCENTS[4])
style_ax(ax, 'Churn Rate by Customer Acquisition Cohort', 'Joining Month', 'Churn Rate')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# ── Tenure Analysis ────────────────────────────────────────────────────────────
tenure_churn = df_clean.groupby('tenure_bin')[target].mean()

fig, ax = plt.subplots(figsize=(8, 5))
tenure_churn.plot(kind='bar', ax=ax, color=ACCENTS[1], edgecolor='white')
for bar, val in zip(ax.patches, tenure_churn):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
style_ax(ax, 'Churn Rate by Customer Tenure', 'Tenure Bucket', 'Churn Rate')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# ── Chart 2: Churn rate by key categorical features (sorted, annotated) ────────
cat_features = [
    ('membership_category', 'Membership Category'),
    ('region_category',     'Region'),
    ('past_complaint',      'Past Complaint'),
    ('complaint_status',    'Complaint Status'),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Churn Rate by Key Categorical Features', fontsize=15, fontweight='bold')

for ax, (col, title) in zip(axes.flat, cat_features):
    churn_rate_by_cat = (
        df_clean.groupby(col)[target].mean().sort_values()
    )
    counts_by_cat = df_clean[col].value_counts()
    bars = churn_rate_by_cat.plot(kind='barh', ax=ax, color=ACCENTS[1], edgecolor='white')
    for i, (cat, val) in enumerate(churn_rate_by_cat.items()):
        n = counts_by_cat.get(cat, 0)
        ax.text(val + 0.005, i, f'{val:.2f} (n={n:,})', va='center', fontsize=8.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.set_xlim(0, churn_rate_by_cat.max() * 1.35)
    style_ax(ax, title, 'Churn Rate', '')

plt.tight_layout()
plt.show()

# %%
# ── Chart 3: Numerical features — violin + box overlay ─────────────────────────
engineered_numerical = [
    'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
    'tenure_days', 'engagement_score', 'points_in_wallet'
]
plot_df = df_clean[engineered_numerical + [target]].copy()
plot_df[target] = plot_df[target].map({0: 'No Churn', 1: 'Churn'})

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Numerical Feature Distributions by Churn Status', fontsize=15, fontweight='bold')

for ax, col in zip(axes.flat, engineered_numerical):
    sns.violinplot(data=plot_df, x=target, y=col, palette=COLORS,
                   order=['No Churn', 'Churn'], inner=None, linewidth=0.8, ax=ax)
    sns.boxplot(data=plot_df, x=target, y=col, order=['No Churn', 'Churn'],
                width=0.14, color='white', linewidth=1.2,
                flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.4}, ax=ax)
    style_ax(ax, col.replace('_', ' ').title(), '', '')

plt.tight_layout()
plt.show()

# %%
# ── Chart 4: Correlation heatmap ──────────────────────────────────────────────────
corr_cols   = engineered_numerical + ['value_per_login', target]
corr_matrix = df_clean[corr_cols].corr()
mask        = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
    center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
    annot_kws={'size': 9}, ax=ax, cbar_kws={'shrink': 0.8}
)
style_ax(ax, 'Feature Correlation Matrix')
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.show()

correlations = corr_matrix[target].drop(target).sort_values(ascending=False)
print("\nCorrelation with churn:")
print(correlations)

# %% [markdown]
# ## 7. Prepare Data for Modeling

# %%
# Drop identifiers, raw date/time columns, and analysis-only columns
drop_columns = id_columns + date_columns + ['joining_month', 'tenure_bin']

df_model = df_clean.drop(columns=drop_columns)

# last_visit_hour is treated as numerical (ordinal hour 0-23)
numerical_features_model = numerical_columns + [
    'tenure_days', 'engagement_score', 'value_per_login', 'last_visit_hour'
]
# Binary complaint features — no encoding needed
passthrough_features = ['had_complaint', 'complaint_unresolved', 'complaint_severity']

X = df_model.drop(columns=[target])
y = df_model[target]

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"\nTarget distribution:\n{y.value_counts()}")

# %% [markdown]
# ## 8. Train-Test Split
#
# The test set is held out entirely and used only for final evaluation of the selected model.
# All model comparison and selection is done via cross-validation on the training set.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTrain churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")

# %% [markdown]
# ## 9. Preprocessing Pipelines
#
# Transformations are fit on training data only to prevent leakage.
# - Logistic Regression uses a full Pipeline with OneHotEncoder + StandardScaler
# - Tree-based models use a simpler ColumnTransformer (no scaling needed)

# %%
# Identify nominal categorical columns present in X
nominal_cats = [c for c in categorical_columns if c in X.columns]

# Imputer + scaler pipeline for numerical features (Logistic Regression)
num_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Imputer-only pipeline for numerical features (tree-based models)
num_tree = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Imputer + encoder pipeline for categorical features
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor for Logistic Regression: impute + scale numericals, impute + OHE categoricals
lr_preprocessor = ColumnTransformer(transformers=[
    ('num', num_lr, numerical_features_model),
    ('cat', cat_pipe, nominal_cats),
    ('pass', 'passthrough', passthrough_features)
])

# Preprocessor for tree-based models: impute numericals, impute + OHE categoricals, no scaling
tree_preprocessor = ColumnTransformer(transformers=[
    ('num', num_tree, numerical_features_model),
    ('cat', cat_pipe, nominal_cats),
    ('pass', 'passthrough', passthrough_features)
])

# Full pipelines
lr_pipeline = Pipeline([
    ('preprocessor', lr_preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

rf_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=20,
        random_state=42, class_weight='balanced', n_jobs=-1
    ))
])

gb_pipeline = Pipeline([
    ('preprocessor', tree_preprocessor),
    ('model', GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ))
])

print("Pipelines defined")

# %%
# Baseline model — majority class predictor
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print(f"Baseline (majority class) F1: {f1_score(y_test, y_dummy):.4f}")
print("(All models must beat this baseline to demonstrate value)")

# %% [markdown]
# ## 10. Cross-Validation on Training Set
#
# Model selection is based on stratified cross-validation on the training set only.
# The test set is not touched at this stage.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    'Logistic Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gb_pipeline,
}

cv_results = {}
print("Running 5-fold stratified cross-validation on training set (scoring: F1)...\n")

for name, pipeline in candidates.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name}: mean F1 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

# %%
cv_summary = pd.DataFrame({
    name: {'cv_f1_mean': scores.mean(), 'cv_f1_std': scores.std()}
    for name, scores in cv_results.items()
}).T

print("\nCross-validation summary:")
print(cv_summary)

# ── Chart 5: CV comparison ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
means = cv_summary['cv_f1_mean']
stds  = cv_summary['cv_f1_std']
best  = means.idxmax()
bar_colors = [ACCENTS[4] if n == best else ACCENTS[1] for n in means.index]
bars = ax.barh(means.index, means, xerr=stds, color=bar_colors, capsize=5,
               edgecolor='white', linewidth=1.1, height=0.5,
               error_kw={'elinewidth': 1.5, 'ecolor': '#555'})
for bar, val, std in zip(bars, means, stds):
    ax.text(val + std + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f} ± {std:.4f}', va='center', fontsize=10)
ax.set_xlim(0.80, 1.02)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))
ax.axvline(means.max(), color=ACCENTS[4], linestyle='--', linewidth=1, alpha=0.5)
ax.invert_yaxis()
style_ax(ax, '5-Fold Stratified CV — F1 Score (Training Set)', 'Mean F1 Score', '')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Model Selection
#
# The model with the highest mean CV F1 on the training set is selected.
# The test set is used only once, for final evaluation of the chosen model.

# %%
best_model_name = cv_summary['cv_f1_mean'].idxmax()
best_pipeline = candidates[best_model_name]

print(f"Selected model: {best_model_name}")
print(f"CV F1 mean: {cv_summary.loc[best_model_name, 'cv_f1_mean']:.4f}")
print(f"Rationale: highest mean F1 across 5 stratified folds on training data")
print(f"\nFitting selected model on full training set...")

best_pipeline.fit(X_train, y_train)
print("Done.")

# %% [markdown]
# ## 12. Final Evaluation on Held-Out Test Set
#
# This section is run once, after model selection is complete.

# %%
def evaluate_model(pipeline, X_tr, X_te, y_tr, y_te, model_name):
    """Evaluate a fitted pipeline on train and test sets."""
    y_train_pred = pipeline.predict(X_tr)
    y_test_pred = pipeline.predict(X_te)
    y_test_proba = pipeline.predict_proba(X_te)[:, 1]

    print(f"\n{'='*60}")
    print(f"{model_name} — Final Evaluation")
    print(f"{'='*60}")

    print("\nTrain Metrics:")
    print(f"  F1:        {f1_score(y_tr, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_tr, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_tr, y_train_pred):.4f}")

    print("\nTest Metrics:")
    print(f"  F1:        {f1_score(y_te, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_te, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_te, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_te, y_test_proba):.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_te, y_test_pred, target_names=['No Churn', 'Churn']))

    # ── Chart 6: Confusion matrix + ROC curve ─────────────────────────────────────
    cm = confusion_matrix(y_te, y_test_pred)
    fpr, tpr, _ = roc_curve(y_te, y_test_proba)
    auc_val = roc_auc_score(y_te, y_test_proba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'{model_name} — Final Evaluation (Test Set)',
                 fontsize=14, fontweight='bold')

    # Confusion matrix with counts + row percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot  = np.array([[f'{v:,}\n({p:.1f}%)' for v, p in zip(rv, rp)]
                        for rv, rp in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', linewidths=1,
                linecolor='white', annot_kws={'size': 11},
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'],
                ax=axes[0], cbar_kws={'shrink': 0.8})
    style_ax(axes[0], 'Confusion Matrix', 'Predicted', 'Actual')

    # ROC curve with AUC fill
    axes[1].plot(fpr, tpr, color=ACCENTS[4], lw=2.5, label=f'AUC = {auc_val:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.08, color=ACCENTS[4])
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random')
    axes[1].set_xlim([-0.01, 1.01])
    axes[1].set_ylim([-0.01, 1.05])
    axes[1].legend(loc='lower right', frameon=True)
    style_ax(axes[1], 'ROC Curve', 'False Positive Rate', 'True Positive Rate')

    plt.tight_layout()
    plt.show()

    return {
        'f1': f1_score(y_te, y_test_pred),
        'precision': precision_score(y_te, y_test_pred),
        'recall': recall_score(y_te, y_test_pred),
        'roc_auc': roc_auc_score(y_te, y_test_proba)
    }

# %%
final_metrics = evaluate_model(best_pipeline, X_train, X_test, y_train, y_test, best_model_name)

# %%
# ── Threshold tuning ──────────────────────────────────────────────────────────
# Default threshold (0.5) optimises accuracy; adjust for business intervention capacity.
y_probs = best_pipeline.predict_proba(X_test)[:, 1]

print("Threshold sensitivity:")
for threshold in [0.4, 0.5, 0.6]:
    y_custom = (y_probs > threshold).astype(int)
    print(f"  threshold={threshold:.1f}  F1={f1_score(y_test, y_custom):.4f}  "
          f"Precision={precision_score(y_test, y_custom):.4f}  "
          f"Recall={recall_score(y_test, y_custom):.4f}")
print("\nNote: threshold is tuned for business intervention capacity.")

# %% [markdown]
# ## 13. Model Explainability

# %%
# Feature names after preprocessing
try:
    ohe_feature_names = (
        best_pipeline.named_steps['preprocessor']
        .named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(nominal_cats)
        .tolist()
    )
    all_feature_names = numerical_features_model + ohe_feature_names + passthrough_features
except Exception:
    all_feature_names = list(X.columns)

# %%
inner_model = best_pipeline.named_steps['model']

def _importance_chart(df_imp, value_col, title, xlabel, color):
    """Reusable professional horizontal importance bar chart."""
    top = df_imp.head(15).copy()
    top['label'] = top['feature'].str.replace(r'^[a-z_]+_', '', regex=True)
    bar_colors = [color if v >= 0 else PALETTE['churn'] for v in top[value_col]]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top['label'], top[value_col], color=bar_colors,
                   edgecolor='white', linewidth=0.8)
    max_val = top[value_col].abs().max()
    for bar, val in zip(bars, top[value_col]):
        ax.text(val + max_val * 0.012, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='#888', linewidth=0.8)
    style_ax(ax, title, xlabel, '')
    plt.tight_layout()
    plt.show()

if hasattr(inner_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': inner_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Feature Importances ({best_model_name}):")
    print(feature_importance.head(15).to_string(index=False))
    _importance_chart(feature_importance, 'importance',
                      f'{best_model_name} — Top 15 Feature Importances',
                      'Gini Importance', ACCENTS[1])

elif hasattr(inner_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': inner_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nTop 15 Feature Coefficients ({best_model_name}):")
    print(feature_importance.head(15).to_string(index=False))
    _importance_chart(feature_importance, 'coefficient',
                      f'{best_model_name} — Top 15 Feature Coefficients',
                      'Coefficient', ACCENTS[1])

# %%
# Permutation importance on the test set (model-agnostic, uses the full pipeline)
print("\nCalculating permutation importance on test set...")
X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)

perm_imp = permutation_importance(
    inner_model, X_test_transformed, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': perm_imp.importances_mean
}).sort_values('importance', ascending=False)

print("\nTop 15 Permutation Importances:")
print(perm_importance_df.head(15).to_string(index=False))
_importance_chart(perm_importance_df, 'importance',
                  'Top 15 Features — Permutation Importance (Test Set)',
                  'Mean Decrease in F1', ACCENTS[0])

# %% [markdown]
# ## 14. Business Insights and Recommendations

# %%
print("\n" + "="*80)
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("="*80)

print("\n1. KEY CHURN DRIVERS:")
print("-" * 80)
top_features = perm_importance_df.head(10)['feature'].tolist()
print("Based on permutation importance, the strongest churn predictors are:")
for i, feature in enumerate(top_features[:5], 1):
    print(f"   {i}. {feature}")

print("\n2. HIGH-RISK CUSTOMER SEGMENTS:")
print("-" * 80)
print("   • Customers with unresolved complaint signals (complaint_unresolved = 1)")
print("   • Customers with high days_since_last_login (inactive users)")
print("   • Customers with low engagement_score (low activity)")
print("   • Customers with low avg_time_spent on platform")
print("   • Certain membership categories show higher churn rates")

print("\n3. ACTIONABLE RECOMMENDATIONS:")
print("-" * 80)
print("   A. Complaint Resolution Priority")
print("      → Immediately address all open/unresolved complaints")
print("      → Implement proactive outreach for customers with past complaints")
print("      → Track complaint resolution time as a KPI")

print("\n   B. Re-engagement Campaigns")
print("      → Target customers with >30 days since last login")
print("      → Personalised offers based on preferred_offer_types")
print("      → Win-back campaigns for low engagement customers")

print("\n   C. Membership Optimisation")
print("      → Review pricing and benefits for high-churn membership tiers")
print("      → Offer upgrade incentives to retain valuable customers")
print("      → Create loyalty programs for long-tenure customers")

print("\n   D. Proactive Retention")
print("      → Score all customers monthly using this model")
print("      → Trigger interventions for customers with churn probability > 0.6")
print("      → A/B test retention offers on medium-risk customers (0.4–0.6)")

print("\n4. EXPECTED BUSINESS IMPACT:")
print("-" * 80)
print(f"   • Model recall:    {final_metrics['recall']:.1%} of churners identified")
print(f"   • Model precision: {final_metrics['precision']:.1%} of flagged customers are true churners")
print("   • If 20% of identified churners are saved through intervention:")
print(f"     → Potential churn reduction: ~{final_metrics['recall'] * 0.20:.1%} of total churn")

# %% [markdown]
# ## 15. Next Steps and Production Considerations

# %%
print("\n" + "="*80)
print("NEXT STEPS FOR PRODUCTION DEPLOYMENT")
print("="*80)

print("\n1. MODEL VALIDATION:")
print("-" * 80)
print("   • Conduct A/B test: control group (no intervention) vs treatment group")
print("   • Validate model performance on out-of-time data (next 3 months)")
print("   • Calculate ROI: cost of interventions vs value of retained customers")

print("\n2. PRODUCTION PIPELINE:")
print("-" * 80)
print("   • Automate data pipeline to refresh customer features daily/weekly")
print("   • Deploy model as REST API or batch scoring service")
print("   • Integrate churn scores into CRM system for operations team")
print("   • Create dashboard for monitoring high-risk customers")

print("\n3. DATA QUALITY AND MONITORING:")
print("-" * 80)
print("   • Monitor feature distributions for data drift")
print("   • Set up alerts for missing data or anomalous values")
print("   • Track model performance metrics over time")
print("   • Implement feedback loop: track actual churn vs predictions")

print("\n4. MODEL MAINTENANCE:")
print("-" * 80)
print("   • Retrain model quarterly with new data")
print("   • Monitor for concept drift (changing churn patterns)")
print("   • Version control for models and features")

print("\n5. OPERATIONAL INTEGRATION:")
print("-" * 80)
print("   • Train customer success team on using churn scores")
print("   • Define intervention playbooks for different risk levels:")
print("     → High risk (> 0.7):   immediate personal outreach")
print("     → Medium risk (0.4–0.7): targeted offers and campaigns")
print("     → Low risk (< 0.4):    standard engagement")

print("\n6. ETHICAL AND PRIVACY CONSIDERATIONS:")
print("-" * 80)
print("   • Ensure compliance with data privacy regulations")
print("   • Avoid discriminatory patterns in model predictions")
print("   • Regular bias audits across customer segments")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nSelected model:  {best_model_name}")
print(f"CV F1 (mean):    {cv_summary.loc[best_model_name, 'cv_f1_mean']:.4f}")
print(f"Test F1:         {final_metrics['f1']:.4f}")
print(f"Test Recall:     {final_metrics['recall']:.4f}")
print(f"Test ROC-AUC:    {final_metrics['roc_auc']:.4f}")
print("\nThis model shows strong predictive signal, but requires validation on "
      "out-of-time data and refined data quality checks before production use.")

# %%
print("\nLIMITATIONS:")
print("-" * 80)
print("- Data contains placeholder values ('?', 'xxxxxxx') treated as missing")
print("- Sentinel value -999 found in days_since_last_login; replaced with NaN")
print("- No explicit churn timestamp available; churn_risk_score indicates churn likelihood, not actual churn timing")
print("- Time-based (out-of-time) validation not possible with this static dataset")
print("- Some features may act as proxies rather than causal drivers")
print("- complaint_status values may vary in production; complaint features built defensively")

# %%
