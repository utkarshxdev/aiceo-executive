"""
model5_summary.py
Model 5: writes a plain-English performance summary for one employee.

How it works:
  1. The other models do the analysis (score, rating check, anomaly check, comments).
  2. This file collects their results as "facts".
  3. The summary is written from those facts only:
       - With a Claude API key set  -> Claude writes a natural report
       - Without a key (default)    -> a rule-based report is written

Needs these files in the same folder:
  tasks_data.csv, rating_model.pkl, anomaly_model.pkl,
  model1_score.py, model4_sentiment.py

Optional (for the Claude version):
  pip install anthropic
  set ANTHROPIC_API_KEY=your_key   (Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key")
"""

import json
import os

import joblib
import pandas as pd

from model1_score import performance_score
from model4_sentiment import comment_sentiment

CLAUDE_MODEL = "claude-sonnet-5-5"
Z_LIMIT = 2.0

SYSTEM_PROMPT = (
    "You are the AI CEO assistant in an employee performance system. "
    "You receive facts about one employee, produced by analysis models. "
    "Write a short report with: Overall summary, Strengths, Concerns, "
    "Review-quality notes (about the manager's reviews), and Recommendations. "
    "Rules: use ONLY the numbers given, never invent facts. Flags are signals for a "
    "human to look at, not proof of wrongdoing. Do not recommend firing, demotion "
    "or pay changes. End by saying a human manager makes the final decision. "
    "Keep it under 250 words, in simple English."
)


# ---------- Step 1: run the models on all data ----------
def prepare_data(csv_path="tasks_data.csv"):
    """Run all models on a CSV file."""
    df = pd.read_csv(csv_path, parse_dates=["deadline", "completed_date"])
    return prepare_from_df(df)


def prepare_from_df(df):
    """Run the trained models on a dataframe of REVIEWED tasks (CSV or database).
    Needs columns: task_id, employee_id, manager_id, deadline, completed_date,
    days_late, approval_status, rating, comment."""
    df = df.copy()
    m2 = joblib.load("rating_model.pkl")
    m3 = joblib.load("anomaly_model.pkl")

    # Same features Model 2 was trained on
    df["is_late"] = (df["days_late"] > 0).astype(int)
    df["status_approved"] = (df["approval_status"] == "approved").astype(int)
    df["status_rejected"] = (df["approval_status"] == "rejected").astype(int)
    df["status_pending"] = (df["approval_status"] == "pending").astype(int)
    df["late_days_only"] = df["days_late"].clip(lower=0)

    df["predicted_rating"] = m2["model"].predict(df[m2["features"]])
    df["gap"] = df["rating"] - df["predicted_rating"]
    df["flagged"] = df["gap"].abs() >= m2["flag_gap"]

    # Model 4: sentiment of each comment (computed once per unique comment, so it is fast)
    comments = df["comment"].fillna("")
    labels = {c: comment_sentiment(c)["label"] for c in comments.unique()}
    df["comment_sentiment"] = comments.map(labels)
    df["comment_mismatch"] = (
        ((df["rating"] >= 4) & (df["comment_sentiment"] == "negative"))
        | ((df["rating"] <= 2) & (df["comment_sentiment"] == "positive"))
    )

    # Manager table (Z-score, same as Model 3)
    mgr = df.groupby("manager_id").agg(
        avg_rating=("rating", "mean"),
        mean_gap=("gap", "mean"),
        pct_five_star=("rating", lambda x: (x == 5).mean()),
    )
    z = (mgr - mgr.mean()) / mgr.std()
    mgr["max_abs_z"] = z.abs().max(axis=1)
    mgr["flagged"] = mgr["max_abs_z"] >= Z_LIMIT

    # Employee anomaly (Isolation Forest from Model 3)
    emp = df.groupby("employee_id").agg(
        mean_gap=("gap", "mean"), gap_std=("gap", "std")
    )
    emp["gap_std"] = emp["gap_std"].fillna(0)
    emp["anomaly"] = m3["isolation_forest"].predict(emp[["mean_gap", "gap_std"]]) == -1

    return df, mgr, emp


# ---------- Step 2: collect facts for one employee ----------
def get_facts(employee_id, data):
    df, mgr, emp = data
    t = df[df["employee_id"] == employee_id].sort_values("deadline")
    if t.empty:
        raise ValueError(f"No tasks found for {employee_id}")

    manager_id = t["manager_id"].iloc[0]
    recent, earlier = t.tail(25), t.iloc[:-25]
    trend = None
    if len(earlier) >= 10:
        trend = round(performance_score(recent) - performance_score(earlier), 1)

    return {
        "employee_id": employee_id,
        "manager_id": manager_id,
        "tasks": int(len(t)),
        "performance_score": performance_score(t),
        "score_change_recent_vs_earlier": trend,
        "on_time_pct": round(float((t["days_late"] <= 0).mean()) * 100, 1),
        "approved_pct": round(float((t["approval_status"] == "approved").mean()) * 100, 1),
        "rejected_pct": round(float((t["approval_status"] == "rejected").mean()) * 100, 1),
        "avg_rating_given": round(float(t["rating"].mean()), 2),
        "avg_rating_expected_from_facts": round(float(t["predicted_rating"].mean()), 2),
        "avg_rating_gap": round(float(t["gap"].mean()), 2),
        "flagged_reviews": int(t["flagged"].sum()),
        "comment_rating_mismatches": int(t["comment_mismatch"].sum()),
        "employee_pattern_unusual": bool(emp.loc[employee_id, "anomaly"]),
        "manager_pattern_unusual": bool(mgr.loc[manager_id, "flagged"]),
        "manager_avg_rating": round(float(mgr.loc[manager_id, "avg_rating"]), 2),
    }


# ---------- Step 3a: rule-based summary (no API needed) ----------
def rule_based_summary(f):
    score = f["performance_score"]
    level = "strong" if score >= 80 else "average" if score >= 60 else "weak"

    article = "an" if level == "average" else "a"
    out = [f"Overall summary: {f['employee_id']} has {article} {level} performance score of "
           f"{score}/100 across {f['tasks']} tasks."]
    if f["score_change_recent_vs_earlier"] is not None:
        c = f["score_change_recent_vs_earlier"]
        direction = "improved" if c > 3 else "dropped" if c < -3 else "stayed steady"
        out[0] += f" Recent performance has {direction} ({c:+} points)."

    strengths, concerns, recs = [], [], []
    if f["on_time_pct"] >= 80:
        strengths.append(f"On-time delivery is good ({f['on_time_pct']}%).")
    else:
        concerns.append(f"Only {f['on_time_pct']}% of tasks were on time.")
        recs.append("Review workload and deadlines; agree on realistic timelines.")
    if f["approved_pct"] >= 80:
        strengths.append(f"{f['approved_pct']}% of tasks were approved.")
    if f["rejected_pct"] >= 20:
        concerns.append(f"{f['rejected_pct']}% of tasks were rejected.")
        recs.append("Add a quality check before submission and give clearer task briefs.")

    notes = []
    gap = f["avg_rating_gap"]
    if gap <= -1:
        notes.append(f"Ratings are about {abs(gap)} stars LOWER than the task facts suggest. "
                     "This may point to unfair or harsh reviewing, and the score above may be understated.")
        recs.append("Ask a second reviewer to re-check a sample of this employee's ratings.")
    elif gap >= 1:
        notes.append(f"Ratings are about {gap} stars HIGHER than the task facts suggest. "
                     "The score may be inflated.")
    if f["flagged_reviews"] > 0:
        notes.append(f"{f['flagged_reviews']} review(s) were flagged as far from the task facts.")
    if f["comment_rating_mismatches"] > 0:
        notes.append(f"{f['comment_rating_mismatches']} review(s) have a comment that "
                     "disagrees with the rating.")
    if f["manager_pattern_unusual"]:
        notes.append(f"Manager {f['manager_id']} has an unusual rating pattern "
                     f"(average {f['manager_avg_rating']}) compared with other managers.")
        recs.append("Ask HR to review this manager's rating habits.")
    if f["employee_pattern_unusual"] and not notes:
        notes.append("This employee's review pattern looks unusual compared with others.")
    if not notes:
        notes.append("No unusual review patterns were found.")
    if not recs:
        recs.append("Keep up the current performance and continue regular check-ins.")

    def block(title, items):
        return f"{title}:\n" + "\n".join(f"  - {i}" for i in (items or ["None found."]))

    return "\n\n".join([
        out[0],
        block("Strengths", strengths),
        block("Concerns", concerns),
        block("Review-quality notes", notes),
        block("Recommendations", recs),
        "Note: These are signals for review, not proof. A human manager makes the final decision.",
    ])


# ---------- Step 3b: Claude summary (optional) ----------
def llm_summary(f):
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(f, indent=2)}],
    )
    return msg.content[0].text


def generate_summary(facts, use_llm=True):
    """Claude if a key is set, otherwise (or if it fails) the rule-based version."""
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            return llm_summary(facts)
        except Exception as e:
            return rule_based_summary(facts) + f"\n\n(Claude was unavailable: {e})"
    return rule_based_summary(facts)


# ---------- Test run ----------
if __name__ == "__main__":
    data = prepare_data()
    df = data[0]

    scores = {e: performance_score(g) for e, g in df.groupby("employee_id")}
    best = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)

    for emp_id in [best, worst, "E065", "E028"]:
        print("=" * 60)
        print(f"EMPLOYEE {emp_id}")
        print("=" * 60)
        print(generate_summary(get_facts(emp_id, data)))
        print()
