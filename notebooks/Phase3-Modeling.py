#!/usr/bin/env python
# coding: utf-8

# # Phase 3 — Machine Learning Modeling
# ## Healthcare AI System
# Input: outputs/model_table.csv — 25,000 rows
# 
# **Output:**
# 
# models/risk_model.joblib
# models/claim_model.joblib
# outputs/feature_schema.json
# 
# **Two Models:**
# 
# Model A — Visit Risk Classification (Low / Medium / High)
# 
# Model B — Claim Outcome Prediction (Paid / Pending / Rejected)

# In[20]:


# Imports
import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             f1_score)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

print("Imports done ✓")


# # Model A — Visit Risk Classification
# **Business Purpose:**
# 
# Predict whether a hospital visit represents Low, Medium, or High operational and clinical risk.
# 
# **Target:** risk_score — Low / Medium / High
# 
# **Split:** Time-based — earliest 80% train, latest 20% test
# 
# **Why time-based?** Prevents data leakage from future visits influencing predictions on past visits.

# In[21]:


# Step 1) Load the dataset
df = pd.read_csv("../outputs/model_table.csv", parse_dates=["registration_date", "visit_date", "billing_date"])
print("Shape:", df.shape)
df.head()


# In[22]:


# Step 2) risk features
risk_target = "risk_score"

risk_features = [
    "age",
    "gender",
    "city",
    "insurance_provider",
    "chronic_flag",
    "department",
    "visit_type",
    "doctor_id",
    "length_of_stay_hours",
    "days_since_registration",
    "visit_frequency",
    "avg_los_per_patient",
    "visit_month",
    "visit_dayofweek"
]
risk_df = df[risk_features + [risk_target, "visit_date"]].copy()
risk_df = risk_df.dropna(subset=[risk_target, "visit_date"])
print("Risk dataset shape: ", risk_df.shape)
risk_df.head()


# In[23]:


# Step 3) Time-Based split
risk_df = risk_df.sort_values("visit_date").reset_index(drop=True)
split_idx = int(len(risk_df)*0.8)

risk_train = risk_df.iloc[:split_idx].copy()
risk_test = risk_df.iloc[split_idx:].copy()

X_train_risk = risk_train[risk_features]
X_test_risk = risk_test[risk_features]

y_train_risk = risk_train[risk_target]
y_test_risk = risk_test[risk_target]

print("Train shape: ", X_train_risk.shape)
print("Test shape: ", X_test_risk.shape)
print("Train Period: ", risk_train["visit_date"].min().date(), "->", risk_train["visit_date"].max().date())
print("Test Period: ", risk_test["visit_date"].min().date(), "->", risk_test["visit_date"].max().date())


# ### Preprocessing Pipeline
# **Two transformers:**
# **Numeric** → SimpleImputer (median strategy)
# 
# **Categorical** → SimpleImputer + OneHotEncoder
# 
# Combined using ColumnTransformer. This entire pipeline gets saved with the model — no separate preprocessing step at inference time.

# In[24]:


# Step 4) Preprocessing Pipeline
risk_numeric_features = [
    "age", "chronic_flag", "length_of_stay_hours",
    "days_since_registration", "visit_frequency",
    "doctor_id", "avg_los_per_patient",
    "visit_month", "visit_dayofweek"
]

risk_categorical_features = [
    "gender", "city", "insurance_provider",
    "department", "visit_type"
]

risk_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

risk_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

risk_preprocessor = ColumnTransformer(transformers=[
    ("num", risk_numeric_transformer,    risk_numeric_features),
    ("cat", risk_categorical_transformer, risk_categorical_features)
])

print("Preprocessing pipeline ready ✓")


# ### Baseline Model: Logistic Regression
# 
# We always start with the simplest model. Logistic Regression gives us a baseline to beat. class_weight='balanced' — handles class imbalance as proven in Phase 2 EDA.

# In[25]:


# Step 5) Logistic Regression Model
risk_baseline_model = Pipeline(steps=[
    ("preprocessor", risk_preprocessor),
    ("classifier", LogisticRegression(
        max_iter = 1000,
        class_weight="balanced"
    ))
])

# Train
risk_baseline_model.fit(X_train_risk, y_train_risk)

# Predict
risk_baseline_pred_train = risk_baseline_model.predict(X_train_risk)
risk_baseline_pred_test =  risk_baseline_model.predict(X_test_risk)

baseline_acc = accuracy_score(y_test_risk, risk_baseline_pred_test)
print("Risk Baseline Train Accuracy :", round(accuracy_score(y_train_risk, risk_baseline_pred_train), 4))
print("Risk Baseline Test Accuracy  :", round(accuracy_score(y_test_risk,  risk_baseline_pred_test), 4))
print("Risk Baseline Test Weighted F1:", round(f1_score(y_test_risk, risk_baseline_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_risk, risk_baseline_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test_risk, risk_baseline_pred_test))


# ## Advanced Model: Random Forest
# 
# Random Forest captures non-linear interactions across patient, department, and visit characteristics.
# 
# Logistic Regression assumes linear relationships. Hospital risk is rarely linear — a 70-year-old chronic ICU patient is not just the sum of those individual risk factors.
# 
# Key parameters:
# 
# class_weight='balanced_subsample' — balances per tree
# 
# max_depth=8 — prevents overfitting
# 
# min_samples_leaf=10 — minimum samples at leaf node
# 
# Example:
# 
# Doctor 1 → High
# 
# Doctor 2 → High
# 
# Doctor 3 → Medium
# 
# Doctor 4 → High
# 
# ... Final vote → High ✅

# In[26]:


# Step 6) Random Forest Model
risk_rf_model = Pipeline(steps=[
    ("preprocessor", risk_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42
    ))
])

risk_rf_model.fit(X_train_risk, y_train_risk)

risk_rf_pred_train = risk_rf_model.predict(X_train_risk)
risk_rf_pred_test  = risk_rf_model.predict(X_test_risk)

rf_acc = accuracy_score(y_test_risk, risk_rf_pred_test)

print("Risk RF Train Accuracy :", round(accuracy_score(y_train_risk, risk_rf_pred_train), 4))
print("Risk RF Test Accuracy  :", round(accuracy_score(y_test_risk,  risk_rf_pred_test),  4))
print("Risk RF Test Weighted F1:", round(f1_score(y_test_risk, risk_rf_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_risk, risk_rf_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test_risk, risk_rf_pred_test))


# ### XGBoost Model
# 
# **XGBoost uses gradient boosting** — builds trees sequentially, each correcting the errors of the previous one.
# 
# Why XGBoost here:
# 
#     -   Handles mixed numeric + categorical features well
#     -   Built-in regularisation prevents overfitting
#     -   Typically outperforms Random Forest on structured tabular data
# **Note:** XGBoost requires numeric labels. We use LabelEncoder to convert Low/Medium/High → 0/1/2. We keep separate variables (y_train_risk_xgb / y_test_risk_xgb) so LR and RF evaluations above are not affected.

# In[27]:


# Encode target for XGBoost only
le = LabelEncoder()
risk_df["risk_score_encoded"] = le.fit_transform(risk_df["risk_score"])
print("Label Mapping: ", dict(zip(le.classes_, le.transform(le.classes_))))


# In[28]:


# Time-based split for XGBoost
# Using same split_idx — consistent with LR and RF
risk_df = risk_df.sort_values("visit_date").reset_index(drop=True)
split_idx = int(len(risk_df) * 0.8)

risk_train_xgb = risk_df.iloc[:split_idx].copy()
risk_test_xgb  = risk_df.iloc[split_idx:].copy()

X_train_risk = risk_train_xgb[risk_features]
X_test_risk  = risk_test_xgb[risk_features]

# Separate XGB label variables — does NOT overwrite y_train_risk
y_train_risk_xgb = risk_train_xgb["risk_score_encoded"]
y_test_risk_xgb  = risk_test_xgb["risk_score_encoded"]

print("XGBoost train shape:", X_train_risk.shape)
print("XGBoost test shape :", X_test_risk.shape)


# In[29]:


# Step 7) XGBoost Model
risk_xgb_model = Pipeline(steps=[
    ("preprocessor", risk_preprocessor),
    ("classifier", XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ))
])

# Train → Predict → Evaluate
# Train
risk_xgb_model.fit(X_train_risk, y_train_risk_xgb)

# Predict Train
risk_xgb_pred_train = risk_xgb_model.predict(X_train_risk)
# Predict Test
risk_xgb_pred_test  = risk_xgb_model.predict(X_test_risk)

# Save for comparison
xgb_acc = accuracy_score(y_test_risk_xgb, risk_xgb_pred_test)

print("Risk XGB Train Accuracy :", round(accuracy_score(y_train_risk_xgb, risk_xgb_pred_train), 4))
print("Risk XGB Test Accuracy  :", round(accuracy_score(y_test_risk_xgb,  risk_xgb_pred_test),  4))
print("Risk XGB Test Weighted F1:", round(f1_score(y_test_risk_xgb, risk_xgb_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_risk_xgb, risk_xgb_pred_test,
                             target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test_risk_xgb, risk_xgb_pred_test))


# In[30]:


# Model Comparison
print("=" * 45)
print("MODEL A — RISK CLASSIFICATION COMPARISON")
print("=" * 45)
print(f"Logistic Regression : {baseline_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")
print(f"XGBoost             : {xgb_acc:.4f}")
print()
print(f"Best model: XGBoost → {max(baseline_acc, rf_acc, xgb_acc):.4f}")


# ### Optional - Hyperparameter Tuning: Random Forest - Not needed at the moment
# We use RandomizedSearchCV to find the best RF parameters. f1_weighted is our scoring metric — not accuracy. This ensures minority classes (High risk) are weighted properly.

# In[31]:


rf_param_grid = {
    "classifier__n_estimators":   [200, 300, 400],
    "classifier__max_depth":      [8, 12, 16, None],
    "classifier__min_samples_split": [5, 10, 20],
    "classifier__min_samples_leaf":  [1, 2, 4]
}

rf_random_search = RandomizedSearchCV(
    risk_rf_model,           # the pipeline to tune
    param_distributions=rf_param_grid,  # values to try
    n_iter=10,               # try only 10 random combinations
    cv=3,                    # 3-fold cross validation
    scoring="f1_weighted",   # optimise for weighted F1
    n_jobs=-1,               # use all CPU cores
    random_state=42          # reproducible random selection
)

rf_random_search.fit(X_train_risk, y_train_risk)

print("Best Parameters:", rf_random_search.best_params_)


# In[32]:


best_rf_model  = rf_random_search.best_estimator_
rf_tuned_pred  = best_rf_model.predict(X_test_risk)

print("Tuned RF Accuracy   :", round(accuracy_score(y_test_risk, rf_tuned_pred), 4))
print("Tuned RF Weighted F1:", round(f1_score(y_test_risk, rf_tuned_pred, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_risk, rf_tuned_pred))


# ## Model B — Claim Outcome Prediction
# **Business Purpose:**
# 
# Predict whether an insurance claim will be Paid, Pending, or Rejected — before submission.
# 
# **Target:** claim_status — Paid / Pending / Rejected
# 
# **Split:** Time-based on billing_date
# 
# 
# **Leakage prevention:**
# approved_amount and payment_days excluded — both are post-outcome variables that are only known AFTER the claim is processed.

# In[33]:


# Step 1) Define Claim Features
claim_target = "claim_status"

claim_features = [
    "age",
    "gender",
    "city",
    "insurance_provider",
    "chronic_flag",
    "department",
    "visit_type",
    "doctor_id",
    "length_of_stay_hours",
    "risk_score",
    "billed_amount",
    "days_since_registration",
    "visit_frequency",
    "avg_los_per_patient",
    "provider_rejection_rate",
    "visit_month",
    "visit_dayofweek",
    "high_cost_visit_flag"
]

claim_df = df[claim_features + [claim_target, "billing_date"]].copy()
claim_df = claim_df.dropna(subset=[claim_target, "billing_date"])
print("Claim Dataset Shape: ", claim_df.shape)
claim_df.head()


# In[34]:


# Step 2) Claim status distribution
print("Claim Status Distribution")
print(claim_df[claim_target].value_counts())
print()
print("Claim Status % Distribution")
print((claim_df[claim_target].value_counts(normalize=True) * 100).round(2))


# ## Step 3) Time-Based Split for Claim Model
# 
# Sorting by billing_date — not visit_date. Billing events happen after visits. We respect that temporal order.

# In[35]:


claim_df = claim_df.sort_values("billing_date").reset_index(drop=True)
split_idx = int(len(claim_df)*0.8)

claim_train = claim_df.iloc[:split_idx].copy()
claim_test = claim_df.iloc[split_idx:].copy()

X_train_claim = claim_train[claim_features]
X_test_claim = claim_test[claim_features]

y_train_claim = claim_train[claim_target]
y_test_claim = claim_test[claim_target]

print("Train shape:", X_train_claim.shape)
print("Test shape :", X_test_claim.shape)
print("Train period:", claim_train["billing_date"].min().date(),
      "→", claim_train["billing_date"].max().date())
print("Test period :", claim_test["billing_date"].min().date(),
      "→", claim_test["billing_date"].max().date())


# ## Step 4) — Preprocessing Pipeline for Claim Model
# 
# Similar to risk model but claim model has additional features:
# 
# risk_score as a categorical input feature
# 
# billed_amount as numeric
# 
# provider_rejection_rate as numeric

# In[36]:


claim_numeric_features = [
    "age", "chronic_flag", "doctor_id",
    "length_of_stay_hours", "billed_amount",
    "days_since_registration", "visit_frequency",
    "avg_los_per_patient", "provider_rejection_rate",
    "visit_month", "visit_dayofweek", "high_cost_visit_flag"
]

claim_categorical_features = [
    "gender", "city", "insurance_provider",
    "department", "visit_type", "risk_score"
]

claim_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

claim_categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

claim_preprocessor = ColumnTransformer(transformers=[
    ("num", claim_numeric_transformer,    claim_numeric_features),
    ("cat", claim_categorical_transformer, claim_categorical_features)
])

print("Claim preprocessing pipeline ready ✓")


# In[37]:


# Step 5) Logistic Regression Model
claim_baseline_model = Pipeline(steps=[
    ("preprocessor", claim_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])
# Train -> Predict -> Evaluate
# Train
claim_baseline_model.fit(X_train_claim, y_train_claim)
# Predict
claim_baseline_pred_train = claim_baseline_model.predict(X_train_claim)
claim_baseline_pred_test = claim_baseline_model.predict(X_test_claim)
# Evaluate
claim_baseline_acc = accuracy_score(y_test_claim, claim_baseline_pred_test)

print("Claim Baseline Train Accuracy :", round(accuracy_score(y_train_claim, claim_baseline_pred_train), 4))
print("Claim Baseline Test Accuracy  :", round(accuracy_score(y_test_claim,  claim_baseline_pred_test), 4))
print("Claim Baseline Test Weighted F1:", round(f1_score(y_test_claim, claim_baseline_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_claim, claim_baseline_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test_claim, claim_baseline_pred_test))


# In[38]:


# Step 6) Random Forest Model
claim_rf_model = Pipeline(steps=[
    ("preprocessor", claim_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_split=8,
        class_weight="balanced",
        random_state=42
    ))
])

# Train -> Predict -> Evaluate
# Train
claim_rf_model.fit(X_train_claim, y_train_claim)
# Predict
claim_rf_pred_train = claim_rf_model.predict(X_train_claim)
claim_rf_pred_test  = claim_rf_model.predict(X_test_claim)
# Evaluate
claim_rf_acc = accuracy_score(y_test_claim, claim_rf_pred_test)

print("Claim RF Train Accuracy :", round(accuracy_score(y_train_claim, claim_rf_pred_train), 4))
print("Claim RF Test Accuracy  :", round(accuracy_score(y_test_claim,  claim_rf_pred_test), 4))
print("Claim RF Test Weighted F1:", round(f1_score(y_test_claim, claim_rf_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_claim, claim_rf_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test_claim, claim_rf_pred_test))


# In[40]:


# XGBoost
# XGBoost needs numeric labels
le_claim = LabelEncoder()
claim_df["claim_status_encoded"] = le_claim.fit_transform(claim_df["claim_status"])
print("Label mapping:", dict(zip(le_claim.classes_, le_claim.transform(le_claim.classes_))))


# In[41]:


# Time based splitting
claim_df = claim_df.sort_values("billing_date").reset_index(drop=True)
split_idx = int(len(claim_df) * 0.8)

claim_train_xgb = claim_df.iloc[:split_idx].copy()
claim_test_xgb  = claim_df.iloc[split_idx:].copy()

X_train_claim = claim_train_xgb[claim_features]
X_test_claim  = claim_test_xgb[claim_features]

# Separate XGB label variables — does NOT overwrite y_train_claim
y_train_claim_xgb = claim_train_xgb["claim_status_encoded"]
y_test_claim_xgb  = claim_test_xgb["claim_status_encoded"]

print("XGBoost train shape:", X_train_claim.shape)
print("XGBoost test shape :", X_test_claim.shape)


# In[42]:


# Step 7) XGBoost Model
claim_xgb_model = Pipeline(steps=[
    ("preprocessor", claim_preprocessor),
    ("classifier", XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ))
])

# Train
claim_xgb_model.fit(X_train_claim, y_train_claim_xgb)

# Predict
claim_xgb_pred_train = claim_xgb_model.predict(X_train_claim)
claim_xgb_pred_test  = claim_xgb_model.predict(X_test_claim)

# Evaluate
claim_xgb_acc = accuracy_score(y_test_claim_xgb, claim_xgb_pred_test)

print("Claim XGB Train Accuracy :", round(accuracy_score(y_train_claim_xgb, claim_xgb_pred_train), 4))
print("Claim XGB Test Accuracy  :", round(accuracy_score(y_test_claim_xgb,  claim_xgb_pred_test),  4))
print("Claim XGB Test Weighted F1:", round(f1_score(y_test_claim_xgb, claim_xgb_pred_test, average="weighted"), 4))
print("\nClassification Report:\n")
print(classification_report(y_test_claim_xgb, claim_xgb_pred_test,
                             target_names=le_claim.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test_claim_xgb, claim_xgb_pred_test))


# In[43]:


print("=" * 45)
print("MODEL B — CLAIM OUTCOME COMPARISON")
print("=" * 45)
print(f"Logistic Regression : {claim_baseline_acc:.4f}")
print(f"Random Forest       : {claim_rf_acc:.4f}")
print(f"XGBoost             : {claim_xgb_acc:.4f}")
print()
models = {
    "Logistic Regression": claim_baseline_acc,
    "Random Forest":       claim_rf_acc,
    "XGBoost":             claim_xgb_acc
}
best = max(models, key=models.get)
print(f"Best model: {best} → {models[best]:.4f}")

### Three Things Different From Model A XGBoost

#le_claim          →  separate LabelEncoder for claim labels
#billing_date      →  split column instead of visit_date
#target_names=le_claim.classes_  →  shows Paid/Pending/Rejected not 0/1/2


# ### Save Model Artifacts
# 
# We save both final models using joblib. At inference time — FastAPI loads these files at startup. No retraining needed per request.

# In[44]:


joblib.dump(risk_rf_model,   "../models/risk_model.joblib")
joblib.dump(claim_rf_model,  "../models/claim_model.joblib")

print("Models saved:")
print("  ✓ ../models/risk_model.joblib")
print("  ✓ ../models/claim_model.joblib")


# ## Save Feature Schema
# 
# The feature schema tells our FastAPI service:
# 
#     -   Which features to expect in the request payload
#     -   Which column is the target
#     -   What split strategy was used
#     
# This is the contract between the model and the API.

# In[ ]:


feature_schema = {
    "risk_model_features"  : risk_features,    # list of 14 features
    "claim_model_features" : claim_features,   # list of 18 features
    "risk_target"          : "risk_score",     # what Model A predicts
    "claim_target"         : "claim_status",   # what Model B predicts
    "risk_time_column"     : "visit_date",     # split column for Model A
    "claim_time_column"    : "billing_date",   # split column for Model B
    "split_strategy"       : "earliest 80 percent train, latest 20 percent test"
}

with open("../outputs/feature_schema.json", "w") as f:
    json.dump(feature_schema, f, indent=4)

print("Feature schema saved ✓")
print("  → ../outputs/feature_schema.json")

