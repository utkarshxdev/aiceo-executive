"""
generate_data.py
Creates fake (synthetic) task data for the AI CEO project.
Output: tasks_data.csv (10,000 rows)
"""

import numpy as np
import pandas as pd

# ---------- Settings ----------
N_EMPLOYEES = 100
N_MANAGERS = 10
N_TASKS = 10000
SEED = 42

# Planted problems (so the models have something to catch)
LAZY_MANAGER = 2      # M03: often gives 5 stars without checking
BIASED_MANAGER = 6    # M07: rates one employee too low
BIASED_EMPLOYEE = 64  # E065: the employee who gets unfairly low ratings
RANDOM_BAD_RATE = 0.07

rng = np.random.default_rng(SEED)

# ---------- Employees and managers ----------
# Each manager has 10 employees (E001-E010 -> M01, E011-E020 -> M02, ...)
employee_ids = [f"E{i + 1:03d}" for i in range(N_EMPLOYEES)]
manager_of_emp = [i // (N_EMPLOYEES // N_MANAGERS) for i in range(N_EMPLOYEES)]

# Safety check: the biased employee must really belong to the biased manager
assert manager_of_emp[BIASED_EMPLOYEE] == BIASED_MANAGER, "Bias setup is wrong"

# Each employee has a skill level (0 to 1). Higher skill = better work
skill = rng.uniform(0.35, 0.95, N_EMPLOYEES)

# ---------- Comment templates ----------
POSITIVE = [
    "Excellent work, delivered well.",
    "Great job, very clean output.",
    "Very good quality, keep it up.",
    "Impressive work and on time.",
]
NEUTRAL = [
    "Average work, needs some improvement.",
    "Okay output, nothing special.",
    "Task done but could be better.",
]
NEGATIVE = [
    "Poor work, many mistakes.",
    "Not acceptable, needs full rework.",
    "Very late and low quality.",
    "Disappointing result.",
]


def comment_for(rating: int) -> str:
    if rating >= 4:
        return str(rng.choice(POSITIVE))
    if rating == 3:
        return str(rng.choice(NEUTRAL))
    return str(rng.choice(NEGATIVE))


# ---------- Generate tasks ----------
rows = []
start_date = pd.Timestamp("2026-01-01")

for t in range(N_TASKS):
    emp = int(rng.integers(0, N_EMPLOYEES))
    mgr = manager_of_emp[emp]
    s = skill[emp]

    deadline = start_date + pd.Timedelta(days=int(rng.integers(0, 290)))

    # Late or on time? Higher skill = more likely on time
    is_late = rng.random() > (0.30 + 0.65 * s)
    if is_late:
        days_late = int(rng.integers(1, 11))
    else:
        days_late = -int(rng.integers(0, 6))  # early or on the day
    completed = deadline + pd.Timedelta(days=days_late)

    # True quality of the work (0 to 1)
    quality = float(np.clip(s + rng.normal(0, 0.15), 0, 1))

    # Fair rating: what the rating should be, judging by the facts
    fair = 1 + 4 * quality - 0.25 * max(days_late, 0)
    fair = int(np.clip(round(fair), 1, 5))

    # Approval status
    if quality < 0.40 or (days_late > 7 and quality < 0.60):
        status = "rejected"
    elif rng.random() < 0.05:
        status = "pending"
    else:
        status = "approved"

    # Normal manager: close to the fair rating (sometimes +-1)
    rating = int(np.clip(fair + rng.choice([-1, 0, 0, 0, 0, 1]), 1, 5))
    comment = comment_for(rating)

    # ---------- Planted bad reviews ----------
    if mgr == LAZY_MANAGER and rng.random() < 0.40:
        rating = 5
        comment = comment_for(rating)

    elif emp == BIASED_EMPLOYEE:
        rating = max(1, rating - 2)
        comment = comment_for(rating)

    elif rng.random() < RANDOM_BAD_RATE:
        if fair <= 3:
            rating = 5                         # bad work, 5 stars
        else:
            rating = int(rng.integers(1, 3))   # good work, 1 or 2 stars
        # Sometimes the comment follows the real quality (mismatch)
        comment = comment_for(fair if rng.random() < 0.5 else rating)

    rows.append(
        {
            "task_id": f"T{t + 1:05d}",
            "employee_id": employee_ids[emp],
            "manager_id": f"M{mgr + 1:02d}",
            "deadline": deadline.date(),
            "completed_date": completed.date(),
            "days_late": days_late,
            "approval_status": status,
            "rating": rating,
            "comment": comment,
            "fair_rating": fair,  # answer key, for checking only
        }
    )

df = pd.DataFrame(rows)

# Suspicious label: rating is 2 or more stars away from the fair rating
df["is_suspicious"] = (df["rating"] - df["fair_rating"]).abs() >= 2

df.to_csv("tasks_data.csv", index=False)

# ---------- Summary ----------
print("Done! tasks_data.csv created.")
print("Total rows:", len(df))
print("Suspicious reviews:", int(df["is_suspicious"].sum()),
      f"({df['is_suspicious'].mean() * 100:.1f}%)")
print("\nApproval status:")
print(df["approval_status"].value_counts())
print("\nAverage rating per manager:")
print(df.groupby("manager_id")["rating"].mean().round(2))
print("\nE065 average rating:", round(df[df.employee_id == "E065"]["rating"].mean(), 2),
      "| E065 fair rating:", round(df[df.employee_id == "E065"]["fair_rating"].mean(), 2))
