"""
setup_demo.py
One command that prepares everything on a NEW machine / server:
  1. creates the database and the demo accounts   (init_db.py)
  2. loads the sample tasks into the database      (load_tasks.py)
  3. trains Model 2 and Model 3 on tasks_data.csv  (train_model2.py, train_model3.py)
  4. creates the training evidence report          (evaluate_models.py)

Run once:   python setup_demo.py
It refuses to run if aiceo.db already exists, so it can never erase live data by accident.
To start over on purpose:  python setup_demo.py --reset
"""

import os
import subprocess
import sys

if os.path.exists("aiceo.db") and "--reset" not in sys.argv:
    raise SystemExit("aiceo.db already exists, so nothing was changed.\n"
                     "Use  python setup_demo.py --reset  only if you want to erase everything and start over.")

for name in ("aiceo.db", "rating_model.pkl", "anomaly_model.pkl", "model_report.json", "training_log.json"):
    if os.path.exists(name):
        os.remove(name)

steps = [
    ("Creating database and demo accounts", "init_db.py"),
    ("Loading sample tasks", "load_tasks.py"),
    ("Training Model 2 (Random Forest)", "train_model2.py"),
    ("Training Model 3 (anomaly detection)", "train_model3.py"),
    ("Creating the training evidence report", "evaluate_models.py"),
]
for i, (label, script) in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] {label} ...")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(f"\nStopped: {script} failed. Read the error above.")

print("\nAll done. The website is ready to start.")
