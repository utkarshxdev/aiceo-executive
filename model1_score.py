import pandas as pd

def performance_score(tasks: pd.DataFrame) -> float:
    on_time = (tasks["completed_date"] <= tasks["deadline"]).mean()
    approved = (tasks["approval_status"] == "approved").mean()
    avg_rating = tasks["rating"].mean() / 5
    not_rejected = 1 - (tasks["approval_status"] == "rejected").mean()

    score = (0.35 * on_time + 0.25 * approved +
             0.25 * avg_rating + 0.15 * not_rejected) * 100
    return round(score, 1)
