"""
load_tasks.py
Creates the tasks table in aiceo.db and fills it with the sample tasks from
tasks_data.csv, so the manager has real records to review.

Run once (after init_db.py):  python load_tasks.py
Running it again resets ALL tasks (and any reviews you saved) back to the sample data.
"""

import os
import sqlite3

import pandas as pd

DB_FILE = "aiceo.db"
CSV_FILE = "tasks_data.csv"
NEWEST_PENDING = 4  # each employee's newest tasks are left "pending" for the manager to review

TITLES = [
    "Weekly report", "Client presentation", "Bug fixes", "Data cleanup",
    "Design mockup", "Code review", "Test cases", "Documentation",
    "Customer follow-up", "Budget sheet", "Research summary", "Process update",
]

if not os.path.exists(DB_FILE):
    raise SystemExit("aiceo.db not found. Run  python init_db.py  first.")
if not os.path.exists(CSV_FILE):
    raise SystemExit("tasks_data.csv not found. Run  python generate_data.py  first.")

df = pd.read_csv(CSV_FILE, parse_dates=["deadline", "completed_date"])

# Old "pending" rows in the sample data already have ratings, so treat them as reviewed
old_pending = df["approval_status"] == "pending"
df.loc[old_pending & (df["rating"] >= 3), "approval_status"] = "approved"
df.loc[old_pending & (df["rating"] < 3), "approval_status"] = "rejected"

# Each employee's newest tasks are waiting for the manager: pending, no rating, no comment
rank = df.groupby("employee_id")["deadline"].rank(method="first", ascending=False)
newest = rank <= NEWEST_PENDING
df["rating"] = df["rating"].astype("object")
df.loc[newest, "approval_status"] = "pending"
df.loc[newest, "rating"] = None
df.loc[newest, "comment"] = ""

conn = sqlite3.connect(DB_FILE)
conn.execute("DROP TABLE IF EXISTS tasks")
conn.execute(
    """
    CREATE TABLE tasks (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id        TEXT NOT NULL UNIQUE,
        employee_id    TEXT NOT NULL,
        manager_id     TEXT NOT NULL,
        title          TEXT NOT NULL,
        description    TEXT NOT NULL DEFAULT '',
        deadline       TEXT NOT NULL,          -- YYYY-MM-DD
        completed_date TEXT NOT NULL,          -- YYYY-MM-DD
        status         TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'approved', 'rejected')),
        rating         INTEGER CHECK (rating BETWEEN 1 AND 5),
        comment        TEXT NOT NULL DEFAULT '',
        reviewed_at    TEXT                    -- when the manager last saved a review
    )
    """
)
conn.execute("CREATE INDEX idx_tasks_manager ON tasks (manager_id)")
conn.execute("CREATE INDEX idx_tasks_employee ON tasks (employee_id)")

rows = []
for r in df.itertuples():
    rating = None if r.rating is None or pd.isna(r.rating) else int(r.rating)
    rows.append((
        r.task_id, r.employee_id, r.manager_id,
        TITLES[int(r.task_id[1:]) % len(TITLES)], "",
        r.deadline.strftime("%Y-%m-%d"), r.completed_date.strftime("%Y-%m-%d"),
        r.approval_status, rating, r.comment if isinstance(r.comment, str) else "",
        None if r.approval_status == "pending" else "2026-10-01 00:00:00",
    ))
conn.executemany(
    "INSERT INTO tasks (task_id, employee_id, manager_id, title, description, deadline, "
    "completed_date, status, rating, comment, reviewed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    rows,
)
conn.commit()

print(f"Loaded {len(rows)} tasks into {DB_FILE}.")
for status, n in conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status"):
    print(f"  {status}: {n}")
conn.close()
