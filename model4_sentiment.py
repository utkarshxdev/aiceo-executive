"""
model4_sentiment.py
Model 4: reads a manager's comment and decides if it is positive, neutral or negative.
Then compares it with the rating to find mismatches.
No training needed. It uses VADER (a ready-made sentiment model) plus a small
list of workplace phrases, because plain VADER misreads phrases like
"needs some improvement" (it sees the word "improvement" as positive).

Install once:  pip install vaderSentiment
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Scores between -NEUTRAL_BAND and +NEUTRAL_BAND count as neutral
NEUTRAL_BAND = 0.20

# Workplace phrases that mean "average, not good, not bad"
NEUTRAL_PHRASES = [
    "average work", "okay output", "nothing special", "could be better",
    "needs some improvement", "needs improvement", "room for improvement",
    "acceptable", "decent",
]
# Extra negative words VADER may miss in a work context
NEGATIVE_BOOST = ["rework", "late", "missed deadline", "incomplete", "sloppy"]
_analyzer.lexicon.update({w: -1.5 for w in NEGATIVE_BOOST if " " not in w})


def comment_sentiment(text: str) -> dict:
    """Returns {'label': 'positive'|'neutral'|'negative', 'score': -1..1}"""
    t = str(text).lower()

    # Rule 1: a "not acceptable" style phrase is negative, not neutral
    negated = any(n in t for n in ("not acceptable", "not good", "not okay"))

    # Rule 2: known neutral workplace phrase (only if nothing strongly negative)
    if not negated and any(p in t for p in NEUTRAL_PHRASES):
        score = _analyzer.polarity_scores(t)["compound"]
        if score > -0.5:  # not very negative overall
            return {"label": "neutral", "score": 0.0}

    # Otherwise let VADER decide
    score = _analyzer.polarity_scores(t)["compound"]
    if score >= NEUTRAL_BAND:
        label = "positive"
    elif score <= -NEUTRAL_BAND:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": round(score, 3)}


def comment_rating_mismatch(rating: int, text: str) -> bool:
    """True if the comment and the rating disagree."""
    label = comment_sentiment(text)["label"]
    if rating >= 4 and label == "negative":
        return True
    if rating <= 2 and label == "positive":
        return True
    return False


def expected_label(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


# This part runs only when you run this file directly (testing)
if __name__ == "__main__":
    print("=== Quick test ===")
    samples = [
        "Excellent work, delivered well.",
        "Average work, needs some improvement.",
        "Okay output, nothing special.",
        "Task done but could be better.",
        "Poor work, many mistakes.",
        "Not acceptable, needs full rework.",
        "Very late and low quality.",
        "Disappointing result.",
        "Impressive work and on time.",
        "Great job, very clean output.",
    ]
    for s in samples:
        print(f"{comment_sentiment(s)['label']:9} | {s}")

    print("\n=== Scan all reviews ===")
    df = pd.read_csv("tasks_data.csv")
    df["sentiment"] = df["comment"].apply(lambda c: comment_sentiment(c)["label"])
    df["mismatch"] = df.apply(
        lambda r: comment_rating_mismatch(r["rating"], r["comment"]), axis=1
    )

    print("Sentiment counts:")
    print(df["sentiment"].value_counts())

    n = int(df["mismatch"].sum())
    print(f"\nComment vs rating mismatches: {n} ({n / len(df) * 100:.1f}%)")

    # Accuracy check: on normal (non-suspicious) reviews, the comment should
    # match the rating, so we can measure how often the model agrees.
    normal = df[~df["is_suspicious"]].copy()
    normal["expected"] = normal["rating"].apply(expected_label)
    acc = (normal["sentiment"] == normal["expected"]).mean()
    print(f"\nSentiment accuracy on normal reviews: {acc * 100:.1f}%")
    print("\nExpected (rows) vs model (columns):")
    print(pd.crosstab(normal["expected"], normal["sentiment"]))

    # Of the real mismatches planted in the data, how many were caught?
    sus = df[df["is_suspicious"]]
    print(f"\nSuspicious reviews caught by comment check: "
          f"{int(sus['mismatch'].sum())} of {len(sus)}")

    print("\nExamples of mismatches:")
    print(df[df["mismatch"]][["task_id", "rating", "comment", "sentiment"]].head(5))
