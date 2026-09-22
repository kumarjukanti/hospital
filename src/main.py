
# ======================================
# Notebook: Phase4_Mlflow.ipynb
# ======================================

# %% [markdown]
# "MLflow solves this in three ways."
# 
# 
# "First — Experiment Tracking. Every model run gets logged automatically. Parameters, metrics, artifacts. Timestamped. Named. Searchable. You never lose a result again."
# 
# 
# "Second — Model Registry. Your best model gets registered with a version number. Version 1 goes to Staging. After validation it gets promoted to Production. Your serving layer always knows which version is live."
# 
# 
# "Third — Reproducibility. Any run can be re-executed exactly. Same parameters, same data version, same result. That's the production guarantee."

# %% [markdown]
# ```text
# # WITHOUT MLflow                         WITH MLflow
# # ______________                        _____________
# # print("accuracy: 0.95")                mlflow.log_metric("accuracy", 0.95)
# # Where did I save this?            →   stored, searchable, versioned
# # What params did I use?            →   params logged automatically
# # Which was the best run?           →   UI comparison in one click
# # Can I reproduce run 3?            →   yes, run ID captures everything
# ```

# %% [markdown]
# # Phase 4 — MLflow Experiment Tracking
# 
# ## Healthcare AI System
# 
# **Prerequisites:**
# 
# - MLflow UI running at [http://127.0.0.1:5000](http://127.0.0.1:5000)
# - `model_table.csv` in `outputs/` folder
# 
# **Run in terminal before this notebook:**
# 
# ```bash
# cd Healthcare
# mlflow ui
# mlflow server --port 5000

# %%
import mlflow
import joblib
import pandas as pd
import warnings
import json
import os
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
from sklearn.metrics import (accuracy_score,f1_score,recall_score)  
mlflow.set_tracking_uri("sqlite:///../mlflow.db")  # Set your MLflow tracking server URI here
print("MLflow version:", mlflow.__version__)
print("Tracking URI:", mlflow.get_tracking_uri())


# %%
#set the experiment 
mlflow.set_experiment("healthCare -risk-classification") 
print("Experminet Created") # Set your experiment name here

# %%
#step 1) Load the saved model
risk_rf_model = joblib.load("../models/risk_model.joblib")
claim_rf_model = joblib.load("../models/claim_model.joblib")
print("Models loaded successfully.")
print("Risk Model:", type(risk_rf_model))
print("Claim Model:", type(claim_rf_model))

# %%
#step 2) load the data set
df  = pd.read_csv("../outputs/model_table.csv", parse_dates=['registration_date', 'visit_date', "billing_date"])
print("Data loaded successfully.")
print("Shape:", df.shape)
df.head()

# %%
#Step 3) Load Features schema
with open("../outputs/feature_schema.json", "r") as f:
    schema = json.load(f)

risk_features = schema["risk_model_features"]
claim_features = schema["claim_model_features"]   
risk_target = schema["risk_target"]
claim_target = schema["claim_target"]

print("Schema loaded successfully.")
print("Risk Features:", {len(risk_features) })
print("Claim Features:", {len(claim_features) })


# %%
#Step 4)Create separte datasets 
risk_df = df.copy()
claim_df = df.copy()

# %%
# step 5) Time based split
risk_df = risk_df.sort_values("visit_date").reset_index(drop=True)

split_idx = int(len(risk_df) * 0.8)

risk_train = risk_df.iloc[:split_idx].copy()
risk_test  = risk_df.iloc[split_idx:].copy()

X_train_risk = risk_train[risk_features]
X_test_risk  = risk_test[risk_features]

y_train_risk = risk_train[risk_target]
y_test_risk  = risk_test[risk_target]

print("Risk Train shape:", X_train_risk.shape)
print("Risk Test shape :", X_test_risk.shape)
print("Risk Train period:", risk_train["visit_date"].min().date(), "→", risk_train["visit_date"].max().date())
print("Risk Test period :", risk_test["visit_date"].min().date(), "→", risk_test["visit_date"].max().date())

# %%
# Step 6) Time Based Split - For Claim

claim_df = claim_df.sort_values("billing_date").reset_index(drop=True)

split_idx = int(len(claim_df) * 0.8)

claim_train = claim_df.iloc[:split_idx].copy()
claim_test  = claim_df.iloc[split_idx:].copy()

X_train_claim = claim_train[claim_features]
X_test_claim  = claim_test[claim_features]

y_train_claim = claim_train[claim_target]
y_test_claim  = claim_test[claim_target]

print("Claim Train shape:", X_train_claim.shape)
print("Claim Test shape :", X_test_claim.shape)
print("Claim Train period:", claim_train["billing_date"].min().date(), "→", claim_train["billing_date"].max().date())
print("Claim Test period :", claim_test["billing_date"].min().date(), "→", claim_test["billing_date"].max().date())

# %%
# Step 7) Log Risk Model
with mlflow.start_run(run_name="RandomForest-Risk") as run:
    mlflow.log_params({
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced_subsample",
        "evaluation_data": "risk_test_only",
        "split_strategy": "time_based_80_20",
        "split_column": "visit_date"
    })

    pred_risk = risk_rf_model.predict(X_test_risk)

    # Metrics
    acc_risk = accuracy_score(y_test_risk, pred_risk)
    f1_risk = f1_score(y_test_risk, pred_risk, average="weighted")
    high_recall = recall_score(y_test_risk, pred_risk, labels=["High"], average=None)[0]

    # Logging the Metrics
    mlflow.log_metric("accuracy", acc_risk)
    mlflow.log_metric("weighted_f1", f1_risk)
    mlflow.log_metric("high_risk_recall", high_recall)

    # Logging the Model
    mlflow.sklearn.log_model(
        sk_model=risk_rf_model,
        name="model",
        serialization_format="cloudpickle"
        )

    # Fetching the run id
    risk_run_id = run.info.run_id

    print("RandomForest Risk Model logged ✓")
    print(f"  Accuracy         : {acc_risk:.4f}")
    print(f"  Weighted F1      : {f1_risk:.4f}")
    print(f"  High Risk Recall : {high_recall:.4f}")
    print(f"  Run ID           : {risk_run_id}")

# %% [markdown]
# ## Register the Model
#  Registration creates a named, versioned entry in the MLflow Model Registry.
# 
# Every time you register the same name, the version number increments automatically. v1 → v2 → v3 and so on.

# %%
# Step 1) Register the Risk RF Model in Model Registry
from mlflow import register_model

register_model_name = "HealthRiskRFModel"

model_uri = f"runs:/{risk_run_id}/model"

result = register_model(
    model_uri=model_uri,
    name=register_model_name
)

print("Register Model Name: ", result.name)
print("Registered version: ", result.version)

# %%
# Step 2) Saving the Registered version in a variable
risk_model_version = result.version
print("Risk Model Version Saved: ", risk_model_version)

# %%
# Step 3) Create MLflow Client
from mlflow.tracking import MlflowClient
client = MlflowClient()
print("MLflow Client ready ✓")

# %%
# Step 4) Move Risk Model version to Staging
client.transition_model_version_stage(
    name=register_model_name,
    version=risk_model_version,
    stage="Staging"
)
print(f"Model {register_model_name} version {risk_model_version} moved to staging")

# %%
#  Step 5) Check if model qualifies for Production
if acc_risk >= 0.55 and high_recall >=0.70:
    print("Risk Model is eligible for Production Promotion ✓")
else:
    print("Risk Model is not eligible")

# %%
# Step 6) Promote the Risk Model to Production
client.transition_model_version_stage(
    name=register_model_name,
    version=risk_model_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model {register_model_name} version {risk_model_version} moved to Production ✓")

# %%
# Step 7) Load the Production Risk Model from Registry
import mlflow.sklearn

production_risk_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{register_model_name}/Production"
)
# models:/HealthcareRiskRFModel/Production
print("Production Risk Model Loaded ✓")

# %%
# Step 8) Fetch the current Production version
latest_versions = client.get_latest_versions(
    name=register_model_name,
    stages=["Production"]
)

production_version = latest_versions[0].version if latest_versions else None
print("Current Production Version: ", production_version)

# %%
# Step 9) Run a prediction using the Production Risk Model
pred_risk = production_risk_model.predict(X_test_risk.head(5))
print("Predictions: ", pred_risk)

# %%
# Step 10) Log Prediction with Model Version
import hashlib
from datetime import datetime

def hash_input(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()

# %%
# Step 11) Input Payload
input_payload = {
    "age": 52,
    "gender": "M",                        
    "city": "Bangalore",
    "insurance_provider": "CareOne",       
    "chronic_flag": 1,
    "department": "Cardiology",
    "visit_type": "ER",                    
    "doctor_id": 101,
    "length_of_stay_hours": 48,
    "days_since_registration": 300,
    "visit_frequency": 4,
    "avg_los_per_patient": 36.5,
    "visit_month": 3,
    "visit_dayofweek": 2
}

# %%
# Step 12) Generate Hash
input_hash = hash_input(input_payload)
print("Input Hash: ", input_hash)

# %%
# Step 13) Creating a Log Record 

# Step 1: Define logs directory
BASE_DIR = os.getcwd()  # or project root if running from notebooks
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Step 2: Create logs folder if not exists
os.makedirs(LOG_DIR, exist_ok=True)

# Step 3: Define log file path
LOG_FILE = os.path.join(LOG_DIR, "predictions.log")

# Step 4:
prediction_log = {
    "timestamp": datetime.utcnow().isoformat(),
    "model_name": register_model_name,
    "model_version": production_version,
    "input_hash": input_hash,
    "prediction": str(pred_risk[0])
}

with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(prediction_log) + "\n")

print("Prediction logged with model_version ✓")

