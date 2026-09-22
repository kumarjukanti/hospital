#!/usr/bin/env python
# coding: utf-8

# Phase 1 — SQL Analytics Layer
# Healthcare AI System
# Objective: Load raw hospital CSVs into a relational SQLite database and run operational + financial analytics queries.
# 
# Tables:
# 
# - patients — 5,000 rows
# - visits — 25,000 rows
# - billing — 25,000 rows

# In[1]:


#import 
import os
import pandas as pd
import sqlite3  


# In[2]:


# 1) Load raw CSV Files
patients = pd.read_csv("../data/Corrected-Data/patients.csv")
visits = pd.read_csv("../data/Corrected-Data/visits.csv")
billing = pd.read_csv("../data/Corrected-Data/billing.csv")

print("patients shape: ", patients.shape)
print("visits shape: ", visits.shape)
print("billing shape: ", billing.shape)


# In[3]:


# 2) Create Sqlite db
# Create db folder, if it doesn't exits
os.makedirs("../db", exist_ok=True)

# Connect to sqlite and then create db
conn = sqlite3.connect("../db/hospital.db")
cursor = conn.cursor()
print("Database connected:", conn)


# In[4]:


# 3) - Load dataframes into sqlite tables
# write all three dataframes into sqlite tables
patients.to_sql("patients", conn, if_exists="replace", index=False)
visits.to_sql("visits", conn, if_exists="replace", index=False)
billing.to_sql("billing", conn, if_exists="replace", index=False)

print("All three tables loaded into hospital db")


# In[5]:


# 4) Quick preview of tables
print("=== PATIENTS (first 3 rows) ===")
print(pd.read_sql("select * from patients limit 3", conn).to_string())

print("=== VISITS (first 3 rows) ===")
print(pd.read_sql("select * from visits limit 3", conn).to_string())

print("=== BILLING (first 3 rows) ===")
print(pd.read_sql("select * from billing limit 3", conn).to_string())


# In[6]:


# 5) Department workload analysis
dept_workload = pd.read_sql("""
SELECT 
    department,
    COUNT(visit_id)                             AS total_visits,
    ROUND(AVG(length_of_stay_hours), 2)         AS avg_los_hours,
    ROUND(MAX(length_of_stay_hours), 2)         AS max_los_hours,
    SUM(CASE WHEN risk_score = 'High'
        THEN 1 ELSE 0 END)                      AS high_risk_visits
    FROM visits
    GROUP BY department
    ORDER BY total_visits DESC
""", conn)

print(dept_workload.to_string(index=False))


# In[7]:


# 6) Doctor Risk Cases
# Which doctors handle the most high-risk patients?

doctor_risk_cases = pd.read_sql(""" 
SELECT 
    doctor_id,  
    COUNT(visit_id)                                         AS total_visits,
    SUM(CASE WHEN risk_score = 'High' 
        THEN 1 ELSE 0 
        END)                                                AS high_risk_cases,
    ROUND(
        100.0 * SUM(CASE WHEN risk_score = 'High' 
        THEN 1 ELSE 0 END) / COUNT(visit_id), 1)            AS high_risk_pct
FROM visits
GROUP BY doctor_id  
ORDER BY high_risk_cases DESC
LIMIT 10
""", conn)

print(doctor_risk_cases.to_string(index=False))


# In[8]:


# 7) Patient Visit Patterns
# How many visits does each patient make on average?

visit_patterns = pd.read_sql("""
SELECT 
    visit_frequency_bucket,
    COUNT(patient_id) AS num_patients
FROM (
    SELECT 
        patient_id,
        COUNT(visit_id) AS visit_count,
        CASE 
            WHEN COUNT(visit_id) = 1 THEN '1 visit'
            WHEN COUNT(visit_id) BETWEEN 2 AND 3 THEN '2-3 visits'
            WHEN COUNT(visit_id) BETWEEN 4 AND 5 THEN '4-5 visits'
            ELSE '6+ visits' 
        END AS visit_frequency_bucket
    FROM visits
    GROUP BY patient_id
) 
GROUP BY visit_frequency_bucket
ORDER BY num_patients DESC
""", conn)

print(visit_patterns.to_string(index=False))


# In[9]:


# 8) Risk Score Distribution by Department
risk_distribution = pd.read_sql("""
SELECT 
    department,
    SUM(CASE WHEN risk_score = 'Low' THEN 1 ELSE 0 END) AS low,
    SUM(CASE WHEN risk_score = 'Medium' THEN 1 ELSE 0 END) AS medium,
    SUM(CASE WHEN risk_score = 'High' THEN 1 ELSE 0 END) AS high,
    COUNT(*) AS total 
FROM visits
GROUP BY department
ORDER BY High DESC
""", conn)
print(risk_distribution.to_string(index=False))


# Financial Analytics

# In[10]:


# 9) Insurance Billing Breakdown
# Revenue and claim outcomes by insurance provider.
insurance_billing = pd.read_sql("""
    SELECT 
        p.insurance_provider,
        COUNT(b.bill_id)                         AS total_claims,
        ROUND(SUM(b.billed_amount), 0)           AS total_billed,
        ROUND(AVG(b.billed_amount), 0)           AS avg_billed,
        ROUND(SUM(b.approved_amount), 0)         AS total_approved,
        SUM(CASE WHEN b.claim_status = 'Paid'     
            THEN 1 ELSE 0 END)                   AS paid,
        SUM(CASE WHEN b.claim_status = 'Pending'  
            THEN 1 ELSE 0 END)                   AS pending,
        SUM(CASE WHEN b.claim_status = 'Rejected' 
            THEN 1 ELSE 0 END)                   AS rejected
    FROM billing b
    JOIN visits  v ON b.visit_id   = v.visit_id
    JOIN patients p ON v.patient_id = p.patient_id
    GROUP BY p.insurance_provider
    ORDER BY total_billed DESC
""", conn)

print("Insurance Billing Breakdown")
print(insurance_billing.to_string(index=False))


# In[11]:


# 10) Claim Rejection Analysis
# Which insurance providers reject the most claims?
rejection_analysis = pd.read_sql("""
    SELECT 
        p.insurance_provider,
        COUNT(b.bill_id)                              AS total_claims,
        SUM(CASE WHEN b.claim_status = 'Rejected' 
            THEN 1 ELSE 0 END)                        AS rejected_claims,
        ROUND(
            100.0 * SUM(CASE WHEN b.claim_status = 'Rejected' 
            THEN 1 ELSE 0 END) / COUNT(b.bill_id), 1) AS rejection_rate_pct
    FROM billing b
    JOIN visits   v ON b.visit_id   = v.visit_id
    JOIN patients p ON v.patient_id = p.patient_id
    GROUP BY p.insurance_provider
    ORDER BY rejection_rate_pct DESC
""", conn)

print("Claim Rejection Rate by Insurance Provider")
print(rejection_analysis.to_string(index=False))


# In[12]:


# 11) Revenue Realization
# How much of what we bill actually gets approved?
revenue_realization = pd.read_sql("""
    SELECT 
        p.insurance_provider,
        ROUND(SUM(b.billed_amount), 0)                AS total_billed,
        ROUND(SUM(b.approved_amount), 0)              AS total_approved,
        ROUND(
            100.0 * SUM(b.approved_amount) / 
            SUM(b.billed_amount), 1)                  AS realization_rate_pct
    FROM billing b
    JOIN visits   v ON b.visit_id   = v.visit_id
    JOIN patients p ON v.patient_id = p.patient_id
    WHERE b.approved_amount IS NOT NULL
    GROUP BY p.insurance_provider
    ORDER BY realization_rate_pct DESC
""", conn)

print("Revenue Realization Rate by Insurance Provider")
print(revenue_realization.to_string(index=False))


# Data Quality

# In[13]:


# 12) Missing Records Check
print("=== MISSING VALUES CHECK ===\n")

for table in ['patients', 'visits', 'billing']:
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        print(f"{table}:")
        print(nulls.to_string())
        print()
    else:
        print(f"{table}: ✓ No missing values\n")


# In[14]:


# 13) Duplicate Patients Check
duplicate_patients = pd.read_sql("""
    SELECT 
        patient_id,
        COUNT(*) AS occurrences
    FROM patients
    GROUP BY patient_id
    HAVING COUNT(*) > 1
""", conn)

if len(duplicate_patients) == 0:
    print("✓ No duplicate patient_ids found")
else:
    print(f"⚠ Found {len(duplicate_patients)} duplicate patient_ids")
    print(duplicate_patients)


# In[15]:


# 14) Orphan Records Check
# Visits without matching patients
orphan_visits = pd.read_sql("""
    SELECT COUNT(*) AS orphan_visits
    FROM visits v
    LEFT JOIN patients p ON v.patient_id = p.patient_id
    WHERE p.patient_id IS NULL
""", conn)

# Billing without matching visits
orphan_billing = pd.read_sql("""
    SELECT COUNT(*) AS orphan_billing
    FROM billing b
    LEFT JOIN visits v ON b.visit_id = v.visit_id
    WHERE v.visit_id IS NULL
""", conn)

print("Orphan visits  (no matching patient):", 
      orphan_visits['orphan_visits'][0])
print("Orphan billing (no matching visit)  :", 
      orphan_billing['orphan_billing'][0])


# In[16]:


# 15) Invalid LOS Check
invalid_los = pd.read_sql("""
    SELECT 
        COUNT(CASE WHEN length_of_stay_hours <= 0  THEN 1 END) AS zero_or_negative,
        COUNT(CASE WHEN length_of_stay_hours > 720 THEN 1 END) AS over_30_days,
        MIN(length_of_stay_hours)                              AS min_los,
        MAX(length_of_stay_hours)                             AS max_los,
        ROUND(AVG(length_of_stay_hours), 2)                   AS avg_los
    FROM visits
""", conn)

print("LOS Validation:")
print(invalid_los.to_string(index=False))


# In[17]:


# 16) Invalid Payment Days Check
invalid_payment = pd.read_sql("""
    SELECT 
        COUNT(CASE WHEN payment_days < 0   THEN 1 END) AS negative_days,
        COUNT(CASE WHEN payment_days > 365 THEN 1 END) AS over_1_year,
        COUNT(CASE WHEN payment_days IS NULL THEN 1 END) AS nulls,
        MIN(payment_days)                               AS min_days,
        MAX(payment_days)                               AS max_days
    FROM billing
""", conn)

print("Payment Days Validation:")
print(invalid_payment.to_string(index=False))


# Export model_table.csv

# In[18]:


# Join all three tables into one flat table for ML modeling.
# This is the single source of truth for Phase 2 EDA and Phase 3 Modeling.
model_table = pd.read_sql("""
    SELECT
        -- Patient features
        p.patient_id,
        p.age,
        p.gender,
        p.city,
        p.insurance_provider,
        p.chronic_flag,
        p.registration_date,

        -- Visit features
        v.visit_id,
        v.visit_date,
        v.department,
        v.visit_type,
        v.length_of_stay_hours,
        v.risk_score,
        v.doctor_id,

        -- Billing features
        b.bill_id,
        b.billed_amount,
        b.approved_amount,
        b.claim_status,
        b.payment_days,
        b.billing_date

    FROM visits v
    JOIN patients p ON v.patient_id = p.patient_id
    JOIN billing  b ON v.visit_id   = b.visit_id
    ORDER BY v.visit_date
""", conn)

print("model_table shape:", model_table.shape)
print("Columns:", model_table.columns.tolist())
model_table.head(3)


# In[19]:


# save the table
import os
os.makedirs("../outputs", exist_ok=True)
model_table.to_csv("../outputs/model_table.csv", index=False)
print(f"model_table.csv saved --> /outputs/model_table.csv")
print(f"Shape: {model_table.shape}")
print(f"Size: {os.path.getsize('../outputs/model_table.csv')/1024:.1f} KB")


# In[20]:


# Close Connection
conn.close()
print("Database connection closed successfully")

