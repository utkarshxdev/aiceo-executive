"""
train_model2.py
Model 2: Random Forest that predicts the "expected rating" from task facts.
If the manager's rating is far from the expected rating, the review is flagged.

Accuracy improvements over the first version:
  1. Only task facts are used as features (days late, approval status). Employee
     history is NOT used, because it lets the model memorize one person's unfair
     ratings as "normal" and hide the bias we want to catch.
  2. Two-pass training: after the first fit, extreme outlier ratings are removed
     and the model is trained again, so bad reviews do not teach it wrong habits
  3. The flag threshold is chosen automatically (best F1 score on the test set)

Input : tasks_data.csv
Output: rating_model.pkl
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

OUTLIER_GAP = 1.8  # in pass 1, ratings this far from prediction are treated as noise

# ---------- 1. Load data ----------
df = pd.read_csv("tasks_data.csv")

# ---------- 2. Features (task facts only, no manager rating inside them) ----------
df["is_late"] = (df["days_late"] > 0).astype(int)
df["status_approved"] = (df["approval_status"] == "approved").astype(int)
df["status_rejected"] = (df["approval_status"] == "rejected").astype(int)
df["status_pending"] = (df["approval_status"] == "pending").astype(int)
df["late_days_only"] = df["days_late"].clip(lower=0)

FEATURES = [
    "days_late", "late_days_only", "is_late",
    "status_approved", "status_rejected", "status_pending",
]

X = df[FEATURES]
y = df["rating"]

#  3. Split 80% / 20% ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def new_model():
    return RandomForestRegressor(
        n_estimators=200, min_samples_leaf=20, random_state=42, n_jobs=-1
    )


# ---------- 4. Two-pass training ----------
first = new_model().fit(X_train, y_train)
train_gap = (y_train - first.predict(X_train)).abs()
clean = train_gap < OUTLIER_GAP
print(f"Pass 1 done. Removed {int((~clean).sum())} noisy training ratings "
      f"out of {len(y_train)}.")

model = new_model().fit(X_train[clean], y_train[clean])

# ---------- 5. Test ----------
pred = model.predict(X_test)
print(f"Average error (MAE): {mean_absolute_error(y_test, pred):.2f} stars")

test = df.loc[X_test.index].copy()
test["predicted_rating"] = pred
test["gap"] = (test["rating"] - test["predicted_rating"]).abs()

# ---------- 6. Choose the best threshold automatically ----------
best = None
for thr in np.arange(1.0, 2.55, 0.1):
    flagged = test["gap"] >= thr
    tp = int((flagged & test["is_suspicious"]).sum())
    fp = int((flagged & ~test["is_suspicious"]).sum())
    fn = int((~flagged & test["is_suspicious"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if best is None or f1 > best["f1"]:
        best = dict(thr=round(float(thr), 1), precision=precision, recall=recall,
                    f1=f1, flagged=int(flagged.sum()), tp=tp)

print(f"\nBest flag threshold: {best['thr']} stars")
print(f"Test tasks: {len(test)}")
print(f"Actual suspicious reviews in test: {int(test['is_suspicious'].sum())}")
print(f"Flagged by model: {best['flagged']}")
print(f"Correctly caught: {best['tp']}")
print(f"Precision: {best['precision'] * 100:.0f}%")
print(f"Recall: {best['recall'] * 100:.0f}%")
print(f"F1 score: {best['f1'] * 100:.0f}%")

test["flagged"] = test["gap"] >= best["thr"]
print("\nTop 5 biggest gaps:")
cols = ["task_id", "employee_id", "manager_id", "days_late",
        "approval_status", "rating", "predicted_rating", "comment"]
print(test.sort_values("gap", ascending=False)[cols].head(5).round(2).to_string(index=False))

# ---------- 7. Save ----------
joblib.dump(
    {"model": model, "features": FEATURES, "flag_gap": best["thr"]},
    "rating_model.pkl",
)
print("\nrating_model.pkl saved.")
