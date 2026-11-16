"""
ceo_data.py
The AI CEO: runs the TRAINED models (rating_model.pkl, anomaly_model.pkl, sentiment,
report writer) on the reviews the managers saved in the database.
The result is cached and only re-computed when the saved reviews change.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime

import pandas as pd

from model5_summary import get_facts, prepare_from_df, rule_based_summary

DB_FILE = os.environ.get("AICEO_DB_PATH", "aiceo.db")
_lock = threading.Lock()
_cache = {}


def employee_name(emp_id):
    return f"Employee {emp_id[1:]}"


def _fingerprint(conn):
    """A cheap summary of the reviews. If it changes, the analysis is re-run."""
    rows = tuple(conn.execute(
        "SELECT COUNT(*), IFNULL(SUM(rating), 0), IFNULL(MAX(reviewed_at), ''), "
        "SUM(status = 'approved'), SUM(status = 'rejected'), IFNULL(SUM(LENGTH(comment)), 0) "
        "FROM tasks"
    ).fetchone())
    # Also re-run if the trained model files were replaced (after retraining)
    mtimes = tuple(os.path.getmtime(f) if os.path.exists(f) else 0
                   for f in ("rating_model.pkl", "anomaly_model.pkl"))
    return rows + mtimes


def _load_reviewed(conn):
    df = pd.read_sql_query(
        "SELECT task_id, employee_id, manager_id, title, deadline, completed_date, "
        "status AS approval_status, rating, comment FROM tasks "
        "WHERE status != 'pending' AND rating IS NOT NULL", conn)
    df["deadline"] = pd.to_datetime(df["deadline"])
    df["completed_date"] = pd.to_datetime(df["completed_date"])
    df["days_late"] = (df["completed_date"] - df["deadline"]).dt.days
    return df


def _reason(r):
    facts = f"{r.approval_status}, " + ("on time" if r.days_late <= 0 else f"{r.days_late} days late")
    if r.flagged:
        return f"Manager gave {int(r.rating)}, task facts suggest about {r.predicted_rating:.1f} ({facts})"
    return f"Comment sounds {r.comment_sentiment} but rating is {int(r.rating)}"


def _flag_rows(df, limit=None):
    sel = df[df["flagged"] | df["comment_mismatch"]].copy()
    sel["abs_gap"] = sel["gap"].abs()
    sel = sel.sort_values("abs_gap", ascending=False)
    if limit:
        sel = sel.head(limit)
    return [dict(
        task_id=r.task_id, employee_id=r.employee_id, employee=employee_name(r.employee_id),
        manager_id=r.manager_id, title=r.title, rating=int(r.rating),
        expected=round(float(r.predicted_rating), 1), comment=r.comment or "",
        status=r.approval_status,
        delivery="On time" if r.days_late <= 0 else f"{r.days_late} days late",
        kind="rating" if r.flagged else "comment", reason=_reason(r),
    ) for r in sel.itertuples()]


def analyze(force=False):
    """Run the AI on all saved reviews (cached until the reviews change)."""
    with _lock:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        try:
            fp = _fingerprint(conn)
            if not force and _cache.get("fp") == fp:
                return _cache["result"]
            df = _load_reviewed(conn)
            pending = conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'").fetchone()[0]
        finally:
            conn.close()

        data = prepare_from_df(df)
        df, mgr, emp = data

        # ----- managers -----
        z = (mgr[["avg_rating", "mean_gap", "pct_five_star"]]
             - mgr[["avg_rating", "mean_gap", "pct_five_star"]].mean()) \
            / mgr[["avg_rating", "mean_gap", "pct_five_star"]].std()
        managers = []
        for mid, r in mgr.iterrows():
            t = df[df["manager_id"] == mid]
            worst = z.loc[mid].abs().idxmax()
            why = {"avg_rating": "average rating", "mean_gap": "ratings vs task facts",
                   "pct_five_star": "share of 5-star ratings"}[worst]
            direction = "higher" if z.loc[mid, worst] > 0 else "lower"
            managers.append(dict(
                id=mid, reviews=int(len(t)), avg_rating=round(float(r.avg_rating), 2),
                mean_gap=round(float(r.mean_gap), 2),
                pct_five_star=round(float(r.pct_five_star) * 100, 1),
                z=round(float(r.max_abs_z), 2), flagged=bool(r.flagged),
                flagged_reviews=int(t["flagged"].sum()),
                why=f"{why} is much {direction} than other managers" if r.flagged else "",
            ))

        # ----- employees -----
        from model1_score import performance_score
        employees = []
        for eid, g in df.groupby("employee_id"):
            e = emp.loc[eid]
            gap = float(e.mean_gap)
            status = "Normal"
            if e.anomaly:
                status = "Possibly over-rated" if gap > 0 else "Possibly under-rated"
            employees.append(dict(
                id=eid, name=employee_name(eid), manager_id=g["manager_id"].iloc[0],
                tasks=int(len(g)), score=float(performance_score(g)),
                avg_rating=round(float(g["rating"].mean()), 2),
                expected=round(float(g["predicted_rating"].mean()), 2),
                mean_gap=round(gap, 2), flagged=int(g["flagged"].sum()),
                mismatches=int(g["comment_mismatch"].sum()),
                anomaly=bool(e.anomaly), status=status,
            ))
        employees.sort(key=lambda e: e["score"], reverse=True)

        result = dict(
            updated=datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            kpi=dict(
                employees=len(employees), reviewed=int(len(df)), pending=int(pending),
                flagged=int(df["flagged"].sum()),
                flagged_pct=round(float(df["flagged"].mean()) * 100, 1),
                mismatches=int(df["comment_mismatch"].sum()),
                managers_flagged=sum(m["flagged"] for m in managers),
                employees_unusual=sum(e["anomaly"] for e in employees),
            ),
            managers=managers, employees=employees,
            flagged_reviews=_flag_rows(df, limit=40),
            data=data,
        )
        _cache["fp"], _cache["result"] = fp, result
        return result


def employee_report(emp_id):
    """AI report + the flagged reviews for one employee."""
    res = analyze()
    df = res["data"][0]
    if emp_id not in set(df["employee_id"]):
        return None
    facts = get_facts(emp_id, res["data"])
    mine = df[df["employee_id"] == emp_id]
    return dict(
        id=emp_id, name=employee_name(emp_id), facts=facts,
        text=rule_based_summary(facts),
        flags=_flag_rows(mine),
        updated=res["updated"],
    )


def model_report():
    """Training evidence written by evaluate_models.py (None if not created yet)."""
    if not os.path.exists("model_report.json"):
        return None
    with open("model_report.json", encoding="utf-8") as f:
        return json.load(f)


def training_log():
    """History of retraining runs (newest first)."""
    if not os.path.exists("training_log.json"):
        return []
    with open("training_log.json", encoding="utf-8") as f:
        return list(reversed(json.load(f)))
