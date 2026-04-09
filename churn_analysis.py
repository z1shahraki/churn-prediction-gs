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
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# %%
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

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
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# %%
print("\nTarget distribution:")
print(df[target].value_counts())
print(f"\nChurn rate: {df[target].mean():.2%}")

# %%
# Inspect last_visit_time and complaint_status before feature engineering
print("\nlast_visit_time sample values:")
print(df['last_visit_time'].head(10).tolist())

print("\ncomplaint_status value counts:")
print(df['complaint_status'].value_counts())

print("\npast_complaint value counts:")
print(df['past_complaint'].value_counts())

# %% [markdown]
# ## 4. Data Preprocessing

# %%
df_clean = df.copy()

# Parse joining_date as a full datetime (used for tenure calculation)
df_clean['joining_date'] = pd.to_datetime(df_clean['joining_date'], errors='coerce')

# NOTE: last_visit_time contains only time-of-day values (HH:MM:SS), not full datetimes.
# Computing days_since_last_visit from this column would be meaningless.
# Instead, we extract last_visit_hour as a proxy for behavioural pattern (e.g. off-hours usage).
df_clean['last_visit_hour'] = pd.to_datetime(
    df_clean['last_visit_time'], format='%H:%M:%S', errors='coerce'
).dt.hour

print("Missing values in last_visit_hour:", df_clean['last_visit_hour'].isnull().sum())

# %% [markdown]
# ## 5. Feature Engineering

# %%
# Reference date for tenure (use max joining_date as proxy for current date)
reference_date = df_clean['joining_date'].max()
df_clean['tenure_days'] = (reference_date - df_clean['joining_date']).dt.days

# Engagement score: product of time spent and login frequency
df_clean['engagement_score'] = (
    df_clean['avg_time_spent'] * df_clean['avg_frequency_login_days']
).fillna(0)

# Transaction value per login day
df_clean['value_per_login'] = (
    df_clean['avg_transaction_value'] / (df_clean['avg_frequency_login_days'] + 1)
)

# complaint_open flag:
# Inspection shows complaint_status has exactly two values:
#   - "Solved in Follow-up"      → complaint was resolved
#   - "No Information Available" → no resolution recorded; treated as unresolved
# We flag a customer as having an open complaint only when they have a past complaint
# AND the status is "No Information Available" (i.e. no resolution on record).
# "Not Applicable" does not appear in this dataset, so we do not include it.
df_clean['complaint_open'] = (
    (df_clean['past_complaint'] == 'Yes') &
    (df_clean['complaint_status'] == 'No Information Available')
).astype(int)

print("Feature engineering complete")
print("\nEngineered features:")
print("- tenure_days: customer lifetime in days from joining_date")
print("- engagement_score: avg_time_spent x avg_frequency_login_days")
print("- value_per_login: avg_transaction_value per login day")
print("- last_visit_hour: hour of day of last visit (time-of-day behavioural signal)")
print("- complaint_open: past complaint with no resolution on record")

print("\ncomplaint_open distribution:")
print(df_clean['complaint_open'].value_counts())

# %% [markdown]
# ## 6. Exploratory Data Analysis

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

df_clean[target].value_counts().plot(kind='bar', ax=ax[0], color=['#2ecc71', '#e74c3c'])
ax[0].set_title('Churn Distribution (Count)')
ax[0].set_xlabel('Churn Risk Score')
ax[0].set_ylabel('Count')
ax[0].set_xticklabels(['No Churn (0)', 'Churn (1)'], rotation=0)

df_clean[target].value_counts(normalize=True).plot(kind='bar', ax=ax[1], color=['#2ecc71', '#e74c3c'])
ax[1].set_title('Churn Distribution (Proportion)')
ax[1].set_xlabel('Churn Risk Score')
ax[1].set_ylabel('Proportion')
ax[1].set_xticklabels(['No Churn (0)', 'Churn (1)'], rotation=0)

plt.tight_layout()
plt.show()

print(f"Class imbalance ratio: {df_clean[target].value_counts()[0] / df_clean[target].value_counts()[1]:.2f}:1")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

pd.crosstab(df_clean['membership_category'], df_clean[target], normalize='index').plot(
    kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c']
)
axes[0, 0].set_title('Churn Rate by Membership Category')
axes[0, 0].set_ylabel('Proportion')
axes[0, 0].legend(['No Churn', 'Churn'])

pd.crosstab(df_clean['region_category'], df_clean[target], normalize='index').plot(
    kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c']
)
axes[0, 1].set_title('Churn Rate by Region')
axes[0, 1].set_ylabel('Proportion')
axes[0, 1].legend(['No Churn', 'Churn'])

pd.crosstab(df_clean['past_complaint'], df_clean[target], normalize='index').plot(
    kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c']
)
axes[1, 0].set_title('Churn Rate by Past Complaint')
axes[1, 0].set_ylabel('Proportion')
axes[1, 0].legend(['No Churn', 'Churn'])

pd.crosstab(df_clean['complaint_status'], df_clean[target], normalize='index').plot(
    kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c']
)
axes[1, 1].set_title('Churn Rate by Complaint Status')
axes[1, 1].set_ylabel('Proportion')
axes[1, 1].legend(['No Churn', 'Churn'])
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
engineered_numerical = [
    'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
    'tenure_days', 'engagement_score', 'points_in_wallet'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for idx, col in enumerate(engineered_numerical):
    row, col_idx = idx // 3, idx % 3
    df_clean.boxplot(column=col, by=target, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'{col} by Churn')
    axes[row, col_idx].set_xlabel('Churn Risk Score')
    plt.sca(axes[row, col_idx])
    plt.xticks([1, 2], ['No Churn', 'Churn'])

plt.suptitle('')
plt.tight_layout()
plt.show()

# %%
correlations = df_clean[engineered_numerical + ['value_per_login', target]].corr()[target].sort_values(ascending=False)
print("\nCorrelation with churn:")
print(correlations)

# %% [markdown]
# ## 7. Prepare Data for Modeling

# %%
# Columns dropped: identifiers (no predictive value, leakage risk) and raw date/time columns
# (joining_date and last_visit_time are replaced by tenure_days and last_visit_hour)
drop_columns = id_columns + date_columns

df_model = df_clean.drop(columns=drop_columns)

# Define feature sets for the ColumnTransformer
# last_visit_hour is treated as numerical (ordinal hour 0-23)
numerical_features_model = numerical_columns + [
    'tenure_days', 'engagement_score', 'value_per_login', 'last_visit_hour'
]
# complaint_open is already binary (0/1), no encoding needed
passthrough_features = ['complaint_open']

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

# Visualise CV results with error bars
fig, ax = plt.subplots(figsize=(8, 5))
means = cv_summary['cv_f1_mean']
stds = cv_summary['cv_f1_std']
ax.bar(means.index, means, yerr=stds, capsize=5, color=['#3498db', '#2ecc71', '#e67e22'])
ax.set_title('Cross-Validation F1 Score (Training Set)')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1)
ax.set_xticklabels(means.index, rotation=15, ha='right')
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

    cm = confusion_matrix(y_te, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'{model_name} — Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return {
        'f1': f1_score(y_te, y_test_pred),
        'precision': precision_score(y_te, y_test_pred),
        'recall': recall_score(y_te, y_test_pred),
        'roc_auc': roc_auc_score(y_te, y_test_proba)
    }

# %%
final_metrics = evaluate_model(best_pipeline, X_train, X_test, y_train, y_test, best_model_name)

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

if hasattr(inner_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': inner_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Feature Importances ({best_model_name}):")
    print(feature_importance.head(15).to_string(index=False))

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['importance'])
    plt.xlabel('Importance')
    plt.title(f'{best_model_name} — Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

elif hasattr(inner_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'coefficient': inner_model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nTop 15 Feature Coefficients ({best_model_name}):")
    print(feature_importance.head(15).to_string(index=False))

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['feature'], feature_importance.head(15)['coefficient'])
    plt.xlabel('Coefficient')
    plt.title(f'{best_model_name} — Top 15 Feature Coefficients')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

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

plt.figure(figsize=(10, 8))
plt.barh(perm_importance_df.head(15)['feature'], perm_importance_df.head(15)['importance'])
plt.xlabel('Permutation Importance')
plt.title('Top 15 Features by Permutation Importance (Test Set)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

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
print("   • Customers with unresolved complaints (complaint_open = 1)")
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
print("\nThis model is ready for business review and pilot testing.")

# %%
