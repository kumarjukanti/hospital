#!/usr/bin/env python
# coding: utf-8

# Phase 2 — Exploratory Data Analysis
# Healthcare AI System
# Input: outputs/model_table.csv — 25,000 rows, 20 columns
# Output: Engineered model_table.csv ready for Phase 3 Modeling
# 
# EDA Philosophy: What Architects Look For
# A data scientist asks: "What does the data look like?"
# An architect asks: "What will this data do to my model?"
# 
# These are four questions every architect answers before touching a model:
# 
# 1. Are the labels meaningful?
# If labels are random, no model can learn them. Period.
# 
# 2. Is there class imbalance?
# Imbalance causes models to ignore minority classes entirely — called class collapse.
# 
# 3. Which features have real signal?
# Correlation tells us where to focus feature engineering effort.
# 
# 4. Are outliers errors or reality?
# In healthcare, extreme values are often the most important rows.
# 
# The answers to these four questions shape every modeling decision we make in Phase 3.
# 

# In[1]:


# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


# In[2]:


# Load Model Tables
df = pd.read_csv("../outputs/model_table.csv")
df.shape


# In[3]:


# Convert date columns 
df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
df["billing_date"] =pd.to_datetime(df["billing_date"], errors="coerce")
df.info()


# In[4]:


df.describe(include="all").T


# 
# # Distribution Analysis
# 
# ### Step 1 - Missing Values Analysis
# ### First thing we always check — what is missing and why.
# ### Not all nulls are data errors. Some are business logic.

# In[5]:


# overall null check
df.isnull().sum().sort_values(ascending=False)


# In[6]:


# Focus on key columns
df[["approved_amount", "payment_days", "length_of_stay_hours"]].isnull().sum()


# ## Step 2 — Business Logic Validation
# ### Three checks before we touch any model.
# We are validating that the data makes business sense — not just statistical sense.

# In[7]:


# Check A - Paid claims should always have an approved amount
# if this returns 0, then we are clean
df[
    (df["claim_status"] == "Paid") &
    (df["approved_amount"].isna())
].shape


# In[8]:


# Check B - payment days missing breakdown by claim status
df[df["payment_days"].isna()]["claim_status"].value_counts()


# In[9]:


# Check C - LOS should never be negative
(df["length_of_stay_hours"]<0).sum()


# # Step 3 — Distribution Analysis
# #### We look at every important column — categorical and numeric.
# #### Goal: understand the shape of our data before building any model.

# In[10]:


# Categorical column reports
print("======= Department================")
print(df["department"].value_counts())
print("\n======Visit Type=================")
print(df["visit_type"].value_counts())
print("\n=======Insurance provider=========")
print(df["insurance_provider"].value_counts())
print("\n=======City=======================")
print(df["city"].value_counts())


# In[11]:


# Barchart showing visits by department
sns.countplot(data=df, x="department", order=df["department"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Visits by Department")
plt.tight_layout()
plt.show()


# In[12]:


# Visits by Type
sns.countplot(data=df, x="visit_type", order=df["visit_type"].value_counts().index)
plt.title("Visits by Type")
plt.show()


# In[13]:


# Visits by Insurance Provider
sns.countplot(data=df, x="insurance_provider", order=df["insurance_provider"].value_counts().index)
plt.title("Visits by Insurance Provider")
plt.show()


# In[14]:


# Visits by City
sns.countplot(data=df, x="city", order=df["city"].value_counts().index)
plt.title("Visits by City")
plt.show()


# In[15]:


# Age Dsitribution
sns.histplot(df["age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()
print(df["age"].describe().round(2))


# In[16]:


# Length of Stay Distribution
sns.histplot(df["length_of_stay_hours"], bins=30, kde=True, color="steelblue")
plt.title("Length of Stay Hours")
plt.xlabel("Hours")
plt.show()
print(df["length_of_stay_hours"].describe().round(2))


# In[17]:


# IQR = Interquartile Range - the spread of the middle 50% of your data
# 25% = 9.96  (bottom quarter boundary)
# 75% = 27.31 (top quarter boundary)
IQR = 27.31 - 9.96 
print(IQR)


# In[18]:


# Upper Fence
# An upper fence is the cutoff, beyond which a value is considered to an outlier
# upper_fence = 75% + (1.5*IQR)
# John's variable = 1.5 (universal multiplier)

upper_fence = 27.31 + (1.5*17.35)
print(upper_fence)


# # Outlier Detection
# ## Step 4 — Outlier Detection
# Two approaches:
# 
# #### Boxplot — visual detection
# ##### IQR method — numerical calculation
# ##### IQR = Q3 − Q1 (middle 50% of data)
# ##### Lower bound = Q1 − 1.5 × IQR
# ##### Upper bound = Q3 + 1.5 × IQR
# 
# Anything outside these bounds = outlier.

# In[19]:


sns.boxplot(x=df["billed_amount"])
plt.title("Billed Amount - Outlier Check")
plt.show()


# In[20]:


sns.boxplot(x=df["payment_days"])
plt.title("Payment Days - Outlier Check")
plt.show()


# In[21]:


sns.boxplot(x=df["length_of_stay_hours"])
plt.title("Length of Stay - Outlier Check")
plt.show()


# In[22]:


# IQR explicit calculation — numbers behind the boxplot
for col in ["length_of_stay_hours", "billed_amount", "payment_days"]:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"{col}")
    print(f"  Q1 = {Q1:.1f}  |  Q3 = {Q3:.1f}  |  IQR = {IQR:.1f}")
    print(f"  Lower bound = {lower:.1f}  |  Upper bound = {upper:.1f}")
    print(f"  Outliers    = {n_out} rows")
    print()


# # Feature Correlations
# **Step 5 — Feature Correlations**
# We encode the target variables numerically so we can measure correlation.
# 
# #### risk_score → Low=0, Medium=1, High=2
# #### claim_status → Paid=0, Pending=1, Rejected=2

# In[23]:


# Encode target for correlation analysis
df["risk_numeric"] = df["risk_score"].map(
    {"Low": 0, "Medium": 1, "High": 2}
)
df["claim_numeric"] = df["claim_status"].map(
    {"Paid": 0, "Pending": 1, "Rejected": 2}
)
print("risk_numeric :",  {"Low": 0, "Medium": 1, "High": 2})
print("claim_numeric :", {"Paid": 0, "Pending": 1, "Rejected": 2})


# In[24]:


 # Full corrleation heatmap
numeric_cols = [
    "age",
    "chronic_flag",
    "length_of_stay_hours",
    "billed_amount",
    "payment_days",
    "risk_numeric",
    "claim_numeric"
]
corr = df[numeric_cols].corr().round(2)
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidth=0.5, square=True)
plt.title("Feature Correlation HeatMap")
plt.tight_layout()
plt.show()


# In[25]:


# What correlates with risk_score?
corr_risk = corr["risk_numeric"].drop("risk_numeric")

print("Correlation with risk_score (sorted):")
print(corr_risk.sort_values(ascending=False).round(3))


# In[26]:


# What correlates with claim_status?
corr_claim = corr["claim_numeric"].drop("claim_numeric")

print("Correlation with claim_status (sorted):")
print(corr_claim.sort_values(ascending=False).round(3))


# In[27]:


# LOS by risk score — the most important relationship
sns.boxplot(data=df, x="risk_score", y="length_of_stay_hours",
            order=["Low", "Medium", "High"])
plt.title("Length of Stay by Risk Score")
plt.show()

print("\nMean LOS by Risk Score:")
print(df.groupby("risk_score")["length_of_stay_hours"]
      .mean().round(2))


# In[28]:


# Rejection rate by insurance provider
df["is_rejected"] = (df["claim_status"] == "Rejected").astype(int)

rej_rate = (df.groupby("insurance_provider")["is_rejected"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
            .round(1))

print("Rejection Rate by Insurance Provider (%):")
print(rej_rate)

rej_rate.plot(kind="bar", color="red", edgecolor="white")
plt.title("Rejection Rate by Insurance Provider")
plt.ylabel("Rejection %")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ## Class Imbalance Deep Dive
# ### Step 6 — Class Imbalance & Class Collapse Demo
# Class imbalance = one class has significantly more rows than others.
# 
# Class collapse = the model ignores minority classes and predicts only the majority class — achieving decent accuracy by being lazy.
# 
# We will prove this live with a before/after demo.

# In[29]:


# Naive baseline - what happens if we always predict majority class

most_common = df["risk_score"].mode()[0]
naive_acc = (df["risk_score"] == most_common).mean()
print(f"Majority class: '{most_common}'") 
print(f"Naive Accuracy: '{naive_acc:.1%}'")


# In[30]:


# Demo setup - two features only, to keep it clean
X_demo = df[["length_of_stay_hours", "chronic_flag"]].fillna(0)
y_demo = df["risk_score"]

# without class weight
model_bad = LogisticRegression(max_iter=1000, random_state=42) # Creating the brain
model_bad.fit(X_demo, y_demo) # Training the brain
pred_bad = model_bad.predict(X_demo) # Using the brain
print("Without class_weight='balanced':")
print("-"*45)
print(classification_report(y_demo, pred_bad))
# (0.00 × 5034 + 0.50 × 12470 + 0.00 × 7496) / 25000 = weighted avg example


# In[31]:


# Demo setup - two features only, to keep it clean
#X_demo = df[["length_of_stay_hours", "chronic_flag"]].fillna(0)
#y_demo = df["risk_score"]

# without class weight
model_good = LogisticRegression(max_iter=1000,class_weight="balanced", random_state=42) # Creating the brain
model_good.fit(X_demo, y_demo) # Training the brain
pred_good = model_good.predict(X_demo) # Using the brain
print("With class_weight='balanced':")
print("-"*45)
print(classification_report(y_demo, pred_good))
# (0.00 × 5034 + 0.50 × 12470 + 0.00 × 7496) / 25000 = weighted avg example


# In[32]:


from sklearn.metrics import recall_score

# Before — random labels (from earlier in notebook)
# After  — clinical labels (current)

recall_high_no_weight  = recall_score(y_demo, pred_bad,
                            labels=["High"], average=None)[0]
recall_high_balanced   = recall_score(y_demo, pred_good,
                            labels=["High"], average=None)[0]

print("HIGH RISK RECALL — NEW CLINICALLY-DERIVED DATA")
print("-" * 45)
print(f"Without class_weight : {recall_high_no_weight:.0%}")
print(f"With class_weight    : {recall_high_balanced:.0%}")
print()
print("COMPARISON — OLD vs NEW DATASET (with class_weight)")
print("-" * 45)
print(f"Old dataset (random labels)  : ~20%  accuracy 37%")
print(f"New dataset (clinical labels): {recall_high_balanced:.0%}  accuracy 67%")
print()
print("Same model. Same 2 features. Labels fixed.")
print("30 percentage point accuracy jump from data quality alone.")


# # Step 7 — Feature Engineering
# We create 7 new features from the existing columns.
# These features encode patient behaviour and provider patterns
# that the raw columns cannot express on their own.
# 
# | Feature | Logic | Why it matters |
# |----------|----------|----------|
# |days_since_registration|	visit_date − registration_date|	Long-term patients behave differently|
# |visit_frequency	|count of visits per patient	|Frequent visitors = higher utilisation|
# |avg_los_per_patient|	mean LOS per patient	|Patient-level health baseline|
# |is_rejected	|1 if Rejected else 0	|Helper for rejection rate|
# |provider_rejection_rate	|mean rejection per insurer|	Encodes insurer behaviour|
# |visit_month	|month from visit_date	|Seasonality signal|
# |visit_dayofweek|	day of week from visit_date	|Operational pattern|
# |high_cost_visit_flag	|billed > 75th percentile|	Flags expensive visits|
# 

# In[33]:


# Days since registration
df["days_since_registration"] = abs(
    df["visit_date"] - df["registration_date"]
).dt.days

df[["patient_id", "visit_date",
    "registration_date", "days_since_registration"]].head()


# In[34]:


# Visit frequency per patient
df["visit_frequency"] = df.groupby("patient_id")["visit_id"].transform("count")

df[["patient_id", "visit_frequency"]].drop_duplicates().head(5)


# In[35]:


# Average LOS per patient
df["avg_los_per_patient"] = df.groupby("patient_id")["length_of_stay_hours"].transform("mean")

df[["patient_id", "length_of_stay_hours",
    "avg_los_per_patient"]].head()


# In[36]:


# Provider rejection rate
# is_rejected already created in Cell 37
df["provider_rejection_rate"] = df.groupby("insurance_provider")["is_rejected"].transform("mean")

print("Rejection rate per provider:")
print(df.groupby("insurance_provider")["provider_rejection_rate"]
      .first().round(3))


# In[37]:


# Time-based features
df["visit_month"]      = df["visit_date"].dt.month
df["visit_dayofweek"]  = df["visit_date"].dt.dayofweek

df[["visit_date", "visit_month", "visit_dayofweek"]].head()



# In[38]:


# High cost visit flag — top 25% of billed amount
high_cost_threshold = df["billed_amount"].quantile(0.75)
df["high_cost_visit_flag"] = (df["billed_amount"] > high_cost_threshold).astype(int)

print(f"High cost threshold (75th percentile): ₹{high_cost_threshold:,.0f}")
print()
print(df["high_cost_visit_flag"].value_counts())


# In[39]:


# Final feature check
new_features = [
    "days_since_registration",
    "visit_frequency",
    "avg_los_per_patient",
    "provider_rejection_rate",
    "visit_month",
    "visit_dayofweek",
    "high_cost_visit_flag"
]

print("New features added:")
for f in new_features:
    print(f"  ✓ {f}")

print(f"\nFinal dataset shape: {df.shape}")


# In[40]:


df[new_features].describe().round(2)


# In[41]:


# Save enriched model_table — ready for Phase 3 Modeling
df.to_csv("../outputs/model_table.csv", index=False)

print("model_table.csv saved ✓")
print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nReady for Phase 3 — Modeling")

