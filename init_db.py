"""
init_db.py
Creates the SQLite database (aiceo.db) with a users table and demo accounts.
Run once:  python init_db.py
(Running it again resets the database.)
"""

import os
import sqlite3

from werkzeug.security import generate_password_hash

DB_FILE = "aiceo.db"
METHOD = "pbkdf2:sha256"  # password hashing method

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
conn.execute(
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        email         TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL CHECK (role IN ('employee', 'manager', 'ceo')),
        employee_id   TEXT,   -- e.g. E001 (links to tasks_data.csv), employees only
        manager_id    TEXT    -- e.g. M01 (a manager's own ID, or the employee's manager)
    )
    """
)

users = [("CEO", "ceo@aiceo.com", "ceo123", "ceo", None, None)]

# 10 managers: manager1@aiceo.com ... manager10@aiceo.com
for m in range(1, 11):
    users.append((f"Manager {m:02d}", f"manager{m}@aiceo.com", "manager123",
                  "manager", None, f"M{m:02d}"))

# 100 employees: emp001@aiceo.com ... emp100@aiceo.com
# Employees E001-E010 belong to M01, E011-E020 to M02, and so on
for e in range(1, 101):
    users.append((f"Employee {e:03d}", f"emp{e:03d}@aiceo.com", "employee123",
                  "employee", f"E{e:03d}", f"M{(e - 1) // 10 + 1:02d}"))

conn.executemany(
    "INSERT INTO users (name, email, password_hash, role, employee_id, manager_id) "
    "VALUES (?, ?, ?, ?, ?, ?)",
    [(n, em, generate_password_hash(pw, method=METHOD), r, eid, mid)
     for n, em, pw, r, eid, mid in users],
)
conn.commit()
conn.close()

print(f"Database created: {DB_FILE} with {len(users)} users.")
print("Demo logins:")
print("  CEO      ceo@aiceo.com       / ceo123")
print("  Manager  manager1@aiceo.com  / manager123")
print("  Employee emp001@aiceo.com    / employee123")
