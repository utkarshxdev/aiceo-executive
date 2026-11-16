"""
generate_all_reports.py
Creates a performance report for EVERY employee.

Output:
  all_reports.txt         - full written report for each employee
  employee_overview.csv   - one row per employee (open in Excel)

Uses the rule-based summary (free, instant). No API key needed.
"""

import pandas as pd

from model5_summary import generate_summary, get_facts, prepare_data

data = prepare_data()
df = data[0]
employee_ids = sorted(df["employee_id"].unique())

facts_list = []
with open("all_reports.txt", "w", encoding="utf-8") as f:
    for emp_id in employee_ids:
        facts = get_facts(emp_id, data)
        facts_list.append(facts)
        f.write("=" * 60 + f"\nEMPLOYEE {emp_id} (Manager {facts['manager_id']})\n"
                + "=" * 60 + "\n")
        f.write(generate_summary(facts, use_llm=False) + "\n\n")

overview = pd.DataFrame(facts_list).sort_values("performance_score", ascending=False)
overview.to_csv("employee_overview.csv", index=False)

print(f"Done! Reports created for {len(employee_ids)} employees.")
print("Files: all_reports.txt, employee_overview.csv\n")

print("Top 5 by performance score:")
print(overview[["employee_id", "manager_id", "performance_score"]].head(5).to_string(index=False))
print("\nBottom 5 by performance score:")
print(overview[["employee_id", "manager_id", "performance_score"]].tail(5).to_string(index=False))

review_issues = overview[
    (overview["avg_rating_gap"].abs() >= 1) | overview["manager_pattern_unusual"]
]
print(f"\nEmployees with review-quality concerns: {len(review_issues)}")
print(review_issues[["employee_id", "manager_id", "avg_rating_gap"]].to_string(index=False))
