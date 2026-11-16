"""
team_data.py
Reads tasks from the database and prepares numbers for the manager pages.
Manager side is manual: no AI here. (The AI analysis lives on the CEO side.)
"""

import os
import sqlite3

import pandas as pd

from model1_score import performance_score

DB_FILE = os.environ.get("AICEO_DB_PATH", "aiceo.db")


def _load(manager_id):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql_query(
            "SELECT task_id, employee_id, title, deadline, completed_date, "
            "status AS approval_status, rating, comment, reviewed_at "
            "FROM tasks WHERE manager_id = ?",
            conn, params=(manager_id,),
        )
    finally:
        conn.close()
    df["deadline"] = pd.to_datetime(df["deadline"])
    df["completed_date"] = pd.to_datetime(df["completed_date"])
    df["days_late"] = (df["completed_date"] - df["deadline"]).dt.days
    df["month"] = df["deadline"].dt.to_period("M")
    return df


def employee_name(emp_id):
    return f"Employee {emp_id[1:]}"


def team_employee_ids(manager_id):
    """Employees of this manager (from the users table, so new employees with no tasks appear too)."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        rows = conn.execute("SELECT employee_id FROM users WHERE role = 'employee' "
                            "AND manager_id = ?", (manager_id,)).fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)


def stats(t):
    """Numbers for any group of tasks (one employee or a whole team)."""
    n = len(t)
    if n == 0:
        return dict(total=0, approved=0, pending=0, rejected=0, completion_pct=0,
                    on_time_pct=0, avg_rating=0, score=0)
    reviewed = t[t["approval_status"] != "pending"]
    approved = int((t["approval_status"] == "approved").sum())
    return dict(
        total=int(n),
        approved=approved,
        pending=int((t["approval_status"] == "pending").sum()),
        rejected=int((t["approval_status"] == "rejected").sum()),
        completion_pct=round(approved / n * 100, 1),
        on_time_pct=round(float((t["days_late"] <= 0).mean()) * 100, 1),
        avg_rating=round(float(reviewed["rating"].mean()), 2) if len(reviewed) else 0,
        score=float(performance_score(reviewed)) if len(reviewed) else 0.0,
    )


def monthly(t):
    rows = []
    for period, g in t.groupby("month"):
        s = stats(g)
        s["month"] = period.strftime("%b %Y")
        rows.append(s)
    return rows


def _overview(team, ids):
    out = []
    for emp_id in ids:
        s = stats(team[team["employee_id"] == emp_id])
        s.update(id=emp_id, name=employee_name(emp_id))
        out.append(s)
    return out


def team_overview(manager_id):
    return _overview(_load(manager_id), team_employee_ids(manager_id))


def employee_stats(manager_id, emp_id):
    t = _load(manager_id)
    s = stats(t[t["employee_id"] == emp_id])
    s.update(id=emp_id, name=employee_name(emp_id))
    return s


def team_stats(manager_id):
    return stats(_load(manager_id))


def employee_tasks(manager_id, emp_id):
    t = _load(manager_id)
    t = t[t["employee_id"] == emp_id].copy()
    t["_pending_first"] = (t["approval_status"] != "pending").astype(int)
    t = t.sort_values(["_pending_first", "deadline"], ascending=[True, False])
    rows = []
    for r in t.itertuples():
        pending = r.approval_status == "pending"
        rows.append(dict(
            task_id=r.task_id,
            title=r.title,
            deadline=r.deadline.strftime("%d %b %Y"),
            finished=r.completed_date.strftime("%d %b %Y"),
            delivery=("On time" if r.days_late <= 0 else f"{r.days_late} days late"),
            late=bool(r.days_late > 0),
            status=r.approval_status,
            rating=None if pd.isna(r.rating) else int(r.rating),
            comment=r.comment or "",
            pending=pending,
        ))
    return rows


def insights(manager_id):
    team = _load(manager_id)
    overview = _overview(team, team_employee_ids(manager_id))
    board = sorted(overview, key=lambda e: e["score"], reverse=True)

    reviewed = team[team["approval_status"] != "pending"]
    rating_counts = [int((reviewed["rating"] == k).sum()) for k in range(1, 6)]

    individual = {}
    for e in overview:
        individual[e["id"]] = dict(
            stats=e, monthly=monthly(team[team["employee_id"] == e["id"]]))

    return dict(
        team=stats(team),
        leaderboard=board,
        status=[sum(e["approved"] for e in overview),
                sum(e["pending"] for e in overview),
                sum(e["rejected"] for e in overview)],
        rating_counts=rating_counts,
        monthly=monthly(team),
        individual=individual,
    )


def reports(manager_id):
    team = _load(manager_id)
    overview = _overview(team, team_employee_ids(manager_id))
    return dict(monthly=monthly(team), completion=overview,
                performance=sorted(overview, key=lambda e: e["score"], reverse=True))
