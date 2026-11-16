"""
retrain_models.py
Retrains Model 2 (Random Forest) and Model 3 (Isolation Forest) on the REVIEWED tasks
saved in the database, so the AI learns from the growing history.

Run from the terminal:   python retrain_models.py
(The CEO "AI Models" page also has a Retrain button that calls the same code.)

Old model files are backed up in the model_backups folder first.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

DB_FILE = os.environ.get("AICEO_DB_PATH", "aiceo.db")
LOG_FILE = "training_log.json"
BACKUP_DIR = "model_backups"
MIN_ROWS = 500
OUTLIER_GAP = 1.8
DEFAULT_FLAG_GAP = 2.0
CONTAMINATION = 0.08
Z_LIMIT = 2.0
FEATURES = ["days_late", "late_days_only", "is_late",
            "status_approved", "status_rejected", "status_pending"]


def _load_reviewed():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql_query(
            "SELECT employee_id, deadline, completed_date, status AS approval_status, rating "
            "FROM tasks WHERE status != 'pending' AND rating IS NOT NULL", conn)
    finally:
        conn.close()
    df["days_late"] = (pd.to_datetime(df["completed_date"]) - pd.to_datetime(df["deadline"])).dt.days
    df["is_late"] = (df["days_late"] > 0).astype(int)
    df["status_approved"] = (df["approval_status"] == "approved").astype(int)
    df["status_rejected"] = (df["approval_status"] == "rejected").astype(int)
    df["status_pending"] = 0  # reviewed tasks are never pending
    df["late_days_only"] = df["days_late"].clip(lower=0)
    return df


def _new_forest():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=42, n_jobs=-1)


def run():
    """Retrain both models. Returns a summary dict (also saved to training_log.json)."""
    df = _load_reviewed()
    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} reviewed tasks found; at least {MIN_ROWS} are needed to retrain.")

    # ---- Model 2: same recipe as train_model2.py, trained on database rows ----
    X, y = df[FEATURES], df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    first = _new_forest().fit(X_train, y_train)
    clean = (y_train - first.predict(X_train)).abs() < OUTLIER_GAP
    model2 = _new_forest().fit(X_train[clean], y_train[clean])
    pred = model2.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))

    # The database has no "answer key", so keep the flag threshold chosen at first training
    flag_gap = DEFAULT_FLAG_GAP
    if os.path.exists("rating_model.pkl"):
        flag_gap = float(joblib.load("rating_model.pkl").get("flag_gap", DEFAULT_FLAG_GAP))
    flagged_pct = float(((y_test - pred).abs() >= flag_gap).mean()) * 100

    # ---- Model 3: Isolation Forest on each employee's average rating gap ----
    df["gap"] = df["rating"] - model2.predict(df[FEATURES])
    emp = df.groupby("employee_id").agg(mean_gap=("gap", "mean"), gap_std=("gap", "std"))
    emp["gap_std"] = emp["gap_std"].fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42)
    iso.fit(emp[["mean_gap", "gap_std"]])

    # ---- Back up the old files, then save the new ones ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for name in ("rating_model.pkl", "anomaly_model.pkl"):
        if os.path.exists(name):
            shutil.copy(name, os.path.join(BACKUP_DIR, f"{stamp}_{name}"))
    joblib.dump({"model": model2, "features": FEATURES, "flag_gap": flag_gap}, "rating_model.pkl")
    joblib.dump({"isolation_forest": iso, "z_limit": Z_LIMIT, "contamination": CONTAMINATION},
                "anomaly_model.pkl")

    summary = {
        "trained_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "source": "database (reviewed tasks)",
        "reviewed_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "noisy_removed": int((~clean).sum()),
        "mae": round(mae, 2),
        "flag_threshold": flag_gap,
        "flagged_pct": round(flagged_pct, 1),
        "employees": int(len(emp)),
    }
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding="utf-8") as f:
            log = json.load(f)
    log.append(summary)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    return summary


if __name__ == "__main__":
    s = run()
    print("Retraining finished.")
    print(f"  Trained on {s['reviewed_rows']} reviewed tasks from the database "
          f"({s['train_rows']} train / {s['test_rows']} test)")
    print(f"  Model 2 average error (MAE): {s['mae']} stars")
    print(f"  Share of test reviews flagged at {s['flag_threshold']} stars: {s['flagged_pct']}%")
    print(f"  Model 3 retrained on {s['employees']} employees")
    print("  Old models backed up in the model_backups folder; log saved to training_log.json")
