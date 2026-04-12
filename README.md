# Customer Churn Analysis

## Overview

This submission analyses customer churn using the provided dataset and covers two connected tasks:

- exploratory analysis to understand customer retention patterns
- predictive modelling to estimate churn risk and support proactive intervention

The aim is not only to build a useful predictive model, but also to generate practical insights that can support business decision-making.

## Selected Model Summary

- **Selected model:** Gradient Boosting
- **Cross-validated F1:** ~0.94
- **Held-out test F1:** ~0.94
- **Main predictive themes:** membership, wallet points, negative feedback, and engagement

## Files Included

- `README.md`, project summary and usage guide
- `churn_analysis.ipynb`, notebook version of the full workflow
- `churn_analysis.py`, Python script version of the same analysis
- `requirements.txt`, project dependencies
- `data/churn.csv`, dataset used by both the notebook and script

---

## Exercise 1, Exploratory Analysis and Business Insights

### Objective

The first part of the work focuses on understanding the dataset, cleaning and preparing it for analysis, and identifying meaningful patterns related to churn and retention.

### What Was Done

The exploratory analysis included:

- cleaning placeholder values and missing entries
- identifying and handling invalid numeric values
- reviewing data structure, distributions, and categorical values
- engineering grouped features to make patterns easier to interpret
- visualising churn patterns across customer segments, value bands, engagement signals, cohorts, and tenure

### Main Findings

The exploratory analysis showed that:

- churn is fairly balanced in this dataset, so standard classification methods are suitable
- grouped features such as login recency, time spent, wallet points, transaction value, and tenure make churn patterns easier to interpret from a business perspective
- some groups matter because their churn rate is higher, while others matter because they contain a large share of customers
- membership category, wallet points, transaction value, and feedback-related fields showed the clearest churn differences
- feedback and membership category show very strong separation in this dataset, so they are useful predictive signals, but they should not be interpreted as direct causes of churn
- transaction value also shows a meaningful pattern, although the exact result depends partly on how value bands are defined
- data-cleaning decisions materially affect the analysis and should remain clearly documented

### Assumptions and Caveats From the Exploratory Analysis

- the supplied target column, `churn_risk_score`, is treated as the churn label
- the working dataset is treated as a binary classification problem
- this is a static historical dataset, not a full event-time churn timeline
- some variables may be recorded close to the churn outcome, so interpretation should remain practical rather than causal

---

## Exercise 2, Predictive Modelling and Advanced Insights

### Objective

The second part of the work focuses on building a predictive model that estimates churn risk, and on identifying signals that can help guide business action.

### Modelling Approach

The modelling workflow included:

- preparation of numeric, categorical, engineered grouped, and flag-based features
- preprocessing within pipelines to keep modelling steps consistent and leakage-safe
- median imputation for numeric features
- a constant `"Missing"` category for categorical features
- one-hot encoding for categorical variables
- stratified train-test split
- stratified cross-validation on the training set
- comparison of multiple model families
- selection of a final model using cross-validated F1
- held-out test evaluation
- feature importance and permutation importance analysis

### Models Compared

**Main comparison models**

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

**Secondary baseline models**

- KNN
- Gaussian Naive Bayes

The secondary baselines were included to show how performance differs across model families, while model selection remained focused on the stronger tabular classifiers.

### Evaluation Approach

The main evaluation metrics were:

- F1 Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

Threshold sensitivity was also reviewed to show how precision and recall change under different decision thresholds. F1 was prioritised because it balances precision and recall for churn identification.

### Main Modelling Results

The modelling results were strong and consistent:

- Gradient Boosting achieved the best cross-validated F1 among the main models
- XGBoost and Random Forest also performed strongly
- Logistic Regression remained a useful benchmark, although tree-based models performed better
- KNN and Gaussian Naive Bayes performed clearly worse, which supports the choice of stronger tabular models
- train-test gaps were small for the leading models, which supports stability on unseen data

**Results snapshot:** Gradient Boosting was selected, with cross-validated F1 around 0.94 and held-out test F1 around 0.94.

### Explainability and Advanced Insights

The strongest predictive themes in the final model were:

- membership category
- customer value and wallet points
- negative feedback
- engagement-related features

These themes appeared in both model-based importance and permutation importance views, which makes the overall pattern more credible within this dataset.

However, these signals should still be treated as predictive rather than causal. Some variables may partly reflect dataset structure or information recorded close to the churn outcome.

---

## Business Recommendations

- follow up quickly with customers who show negative feedback, and review complaint-related signals as supporting context
- re-engage customers who are becoming less active
- test retention offers for lower-value customers and customers with lower wallet points
- review higher-risk membership groups and use different actions for different segments

## How to Test the Model in Practice

- choose a small group of customers for a trial
- use the model to identify customers who appear more likely to leave
- apply a retention action to those customers, such as follow-up support or a targeted offer
- keep another similar group unchanged so the results can be compared fairly
- after the trial period, compare the two groups and check whether fewer customers left in the group that received the action
- also check whether the result was worth the extra effort and cost
- if the result is good, expand the approach to a larger group

## What Is Still Needed Before Production

Before wider operational rollout, the following would still be needed:

- confirm that all required features are available at scoring time in real workflows
- automate data refresh and scoring
- connect outputs to CRM or operational workflows
- monitor drift, missing data, and model performance over time
- validate the model on future data, not only historical holdout data

---

## Key Assumptions and Limitations

- the supplied target is used as given and treated as binary classification
- no true churn timestamp is available, so this is not a full event-time churn model
- `relative_tenure_days` is relative to the latest joining date in the dataset, not true production tenure at scoring time
- some variables, especially complaint and feedback-related fields, may be close to the outcome timing
- feature importance is predictive, not causal
- further validation is required before wider operational use

---

## Environment Setup

Create and activate a virtual environment, then install dependencies.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

### Run the Python Script

```bash
python churn_analysis.py
```

### Run the Notebook

```bash
jupyter notebook churn_analysis.ipynb
```

Then open the notebook and run the cells in order.

## Reproducibility Notes

- the analysis expects the dataset file to be available at `data/churn.csv`
- the notebook and Python script follow the same overall logic and should produce consistent results

## Project Structure

```text
churn-prediction-gs/
├── data/
│   └── churn.csv
├── churn_analysis.py
├── churn_analysis.ipynb
├── requirements.txt
└── README.md
```

## Final Note

This work is intended to provide a practical and interpretable churn-analysis workflow, combining exploratory analysis, predictive modelling, explainability, and business recommendations. The results are useful for prioritising retention actions, while still requiring future-data validation and production-readiness checks before wider use.
