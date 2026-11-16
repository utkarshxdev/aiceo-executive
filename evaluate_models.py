"""
evaluate_models.py
Measures how well your TRAINED models perform and saves the evidence to
model_report.json. The CEO "AI Models" page shows this report.

Run after training:   python evaluate_models.py
Needs: tasks_data.csv, rating_model.pkl, anomaly_model.pkl
"""

import json
import os
import platform
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from model4_sentiment import comment_sentiment

CSV = "tasks_data.csv"
OUTLIER_GAP = 1.8  # same value used in train_model2.py

df = pd.read_csv(CSV, parse_dates=["deadline", "completed_date"])
m2 = joblib.load("rating_model.pkl")
m3 = joblib.load("anomaly_model.pkl")
model2, FEATURES, THR = m2["model"], m2["features"], m2["flag_gap"]

# ---------- Features (same as training) ----------
df["is_late"] = (df["days_late"] > 0).astype(int)
df["status_approved"] = (df["approval_status"] == "approved").astype(int)
df["status_rejected"] = (df["approval_status"] == "rejected").astype(int)
df["status_pending"] = (df["approval_status"] == "pending").astype(int)
df["late_days_only"] = df["days_late"].clip(lower=0)

X, y = df[FEATURES], df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Dataset ----------
dataset = {
    "file": CSV,
    "rows": int(len(df)),
    "employees": int(df["employee_id"].nunique()),
    "managers": int(df["manager_id"].nunique()),
    "date_from": df["deadline"].min().strftime("%d %b %Y"),
    "date_to": df["deadline"].max().strftime("%d %b %Y"),
    "suspicious_reviews": int(df["is_suspicious"].sum()),
    "suspicious_pct": round(float(df["is_suspicious"].mean()) * 100, 1),
    "avg_rating": round(float(df["rating"].mean()), 2),
    "status_counts": {k: int(v) for k, v in df["approval_status"].value_counts().items()},
    "note": "Synthetic (generated) data with deliberately planted bad reviews, "
            "used to train and check the models.",
}

# ---------- Model 2: Random Forest ----------
pred = model2.predict(X_test)
test = df.loc[X_test.index].copy()
test["gap"] = (test["rating"] - pred).abs()
flagged = test["gap"] >= THR
tp = int((flagged & test["is_suspicious"]).sum())
fp = int((flagged & ~test["is_suspicious"]).sum())
fn = int((~flagged & test["is_suspicious"]).sum())
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

first = RandomForestRegressor(n_estimators=model2.n_estimators,
                              min_samples_leaf=model2.min_samples_leaf,
                              random_state=42, n_jobs=-1).fit(X_train, y_train)
removed = int(((y_train - first.predict(X_train)).abs() >= OUTLIER_GAP).sum())

model2_info = {
    "algorithm": "Random Forest Regressor",
    "trees": int(model2.n_estimators),
    "min_samples_leaf": int(model2.min_samples_leaf),
    "features": FEATURES,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "noisy_rows_removed": removed,
    "mae": round(float(np.mean(np.abs(y_test - pred))), 2),
    "flag_threshold": float(THR),
    "precision": round(precision * 100, 1),
    "recall": round(recall * 100, 1),
    "f1": round(f1 * 100, 1),
    "flagged_in_test": int(flagged.sum()),
    "caught": tp,
    "suspicious_in_test": int(test["is_suspicious"].sum()),
    "importance": {k: round(float(v), 4) for k, v in
                   sorted(zip(FEATURES, model2.feature_importances_), key=lambda kv: -kv[1])},
}

# ---------- Model 3: Z-score + Isolation Forest ----------
df["predicted_rating"] = model2.predict(df[FEATURES])
df["gap"] = df["rating"] - df["predicted_rating"]
mgr = df.groupby("manager_id").agg(avg_rating=("rating", "mean"), mean_gap=("gap", "mean"),
                                   pct_five_star=("rating", lambda x: (x == 5).mean()))
z = ((mgr - mgr.mean()) / mgr.std()).abs().max(axis=1)
flagged_managers = sorted(z[z >= m3["z_limit"]].index)

iso = m3["isolation_forest"]
emp = df.groupby("employee_id").agg(mean_gap=("gap", "mean"), gap_std=("gap", "std"))
emp["gap_std"] = emp["gap_std"].fillna(0)
emp["anomaly"] = iso.predict(emp[["mean_gap", "gap_std"]]) == -1
flagged_employees = sorted(emp[emp["anomaly"]].index)

model3_info = {
    "algorithm": "Z-score (managers) + Isolation Forest (employees)",
    "trees": int(iso.n_estimators),
    "contamination": float(m3["contamination"]),
    "z_limit": float(m3["z_limit"]),
    "managers_checked": int(len(mgr)),
    "employees_checked": int(len(emp)),
    "flagged_managers": flagged_managers,
    "flagged_employees": flagged_employees,
    "planted_lazy_manager": "M03",
    "planted_biased_employee": "E065",
    "lazy_manager_found": "M03" in flagged_managers,
    "biased_employee_found": "E065" in flagged_employees,
}

# ---------- Model 4: comment sentiment (VADER + workplace phrases) ----------
labels = {c: comment_sentiment(c)["label"] for c in df["comment"].fillna("").unique()}
sent = df["comment"].fillna("").map(labels)
expected = np.where(df["rating"] >= 4, "positive", np.where(df["rating"] == 3, "neutral", "negative"))
normal = ~df["is_suspicious"]
mismatch = ((df["rating"] >= 4) & (sent == "negative")) | ((df["rating"] <= 2) & (sent == "positive"))
model4_info = {
    "algorithm": "VADER sentiment (pre-trained) + workplace phrase rules",
    "accuracy_normal": round(float((sent[normal] == expected[normal]).mean()) * 100, 1),
    "mismatches": int(mismatch.sum()),
    "suspicious_caught": int((mismatch & df["is_suspicious"]).sum()),
    "suspicious_total": int(df["is_suspicious"].sum()),
    "note": "Comments in the sample data come from fixed templates, so real-world "
            "accuracy will be lower.",
}

report = {
    "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
    "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    "files": {
        f: datetime.fromtimestamp(os.path.getmtime(f)).strftime("%d %b %Y, %H:%M")
        for f in ["rating_model.pkl", "anomaly_model.pkl"]
    },
    "dataset": dataset,
    "model1": {"algorithm": "Weighted scoring formula (no training needed)",
               "weights": {"On-time completion": 35, "Approval rate": 25,
                           "Average rating": 25, "Low rejection rate": 15}},
    "model2": model2_info,
    "model3": model3_info,
    "model4": model4_info,
    "model5": {"algorithm": "Report writer (rule-based; optional Claude API)",
               "note": "Turns the other models' results into plain-English reports."},
}

with open("model_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("model_report.json saved.\n")
print(f"Dataset: {dataset['rows']} tasks, {dataset['employees']} employees, {dataset['managers']} managers")
print(f"Model 2: MAE {model2_info['mae']} | precision {model2_info['precision']}% | "
      f"recall {model2_info['recall']}% | F1 {model2_info['f1']}%")
print(f"Model 3: flagged managers {flagged_managers} | lazy manager found: "
      f"{model3_info['lazy_manager_found']} | biased employee found: {model3_info['biased_employee_found']}")
print(f"Model 4: accuracy on normal reviews {model4_info['accuracy_normal']}%")
