"""
train_model3.py
Model 3: finds unusual patterns.
  Part A: Managers  -> Z-score (compare each manager with the others)
  Part B: Employees -> Isolation Forest (employees whose ratings are far from
                       what Model 2 expected, which can point to bias)

Input : tasks_data.csv, rating_model.pkl (from Model 2)
Output: anomaly_model.pkl
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

Z_LIMIT = 2.0          # a manager is flagged if any Z-score is at least this
CONTAMINATION = 0.08   # about 8% of employees are expected to look unusual

# ---------- 1. Load data and Model 2 ----------
df = pd.read_csv("tasks_data.csv")
saved = joblib.load("rating_model.pkl")
model2 = saved["model"]
FEATURES = saved["features"]

# ---------- 2. Rebuild the same features Model 2 used ----------
df["is_late"] = (df["days_late"] > 0).astype(int)
df["status_approved"] = (df["approval_status"] == "approved").astype(int)
df["status_rejected"] = (df["approval_status"] == "rejected").astype(int)
df["status_pending"] = (df["approval_status"] == "pending").astype(int)
df["late_days_only"] = df["days_late"].clip(lower=0)

# Gap = manager's rating minus the rating Model 2 expected
df["predicted_rating"] = model2.predict(df[FEATURES])
df["gap"] = df["rating"] - df["predicted_rating"]

# ---------- PART A: Managers (Z-score) ----------
mgr = df.groupby("manager_id").agg(
    avg_rating=("rating", "mean"),
    mean_gap=("gap", "mean"),
    pct_five_star=("rating", lambda x: (x == 5).mean()),
)
z = (mgr - mgr.mean()) / mgr.std()
mgr["max_abs_z"] = z.abs().max(axis=1)
mgr["flagged"] = mgr["max_abs_z"] >= Z_LIMIT

print("=== MANAGER CHECK (Z-score) ===")
print(mgr.round(2).sort_values("max_abs_z", ascending=False))
print("\nFlagged managers:", list(mgr[mgr["flagged"]].index))

# ---------- PART B: Employees (Isolation Forest) ----------
emp = df.groupby(["employee_id", "manager_id"]).agg(
    mean_gap=("gap", "mean"),
    gap_std=("gap", "std"),
    tasks=("rating", "size"),
).reset_index()

iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
emp["anomaly"] = iso.fit_predict(emp[["mean_gap", "gap_std"]]) == -1

flagged_emp = emp[emp["anomaly"]].sort_values("mean_gap")

print("\n=== EMPLOYEE CHECK (Isolation Forest) ===")
print(flagged_emp.round(2).to_string(index=False))

if "E065" in set(flagged_emp["employee_id"]):
    print("\nGood: E065 (the planted biased case) was caught.")
else:
    print("\nNote: E065 (the planted biased case) was NOT caught.")

# ---------- Save ----------
joblib.dump(
    {"isolation_forest": iso, "z_limit": Z_LIMIT, "contamination": CONTAMINATION},
    "anomaly_model.pkl",
)
print("\nanomaly_model.pkl saved.")
