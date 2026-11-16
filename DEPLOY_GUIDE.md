# Deploying AI CEO on PythonAnywhere (free plan) for a demo

You will get a link like `https://YOURUSERNAME.pythonanywhere.com` that your teacher can open.
Menu names on the site may look slightly different; the steps stay the same.

## Before you start (important)
- The free plan has a small daily CPU allowance (about 100 CPU-seconds) and small disk (512 MB).
  The setup step below uses roughly 40 CPU-seconds on a test machine, so **run the setup the day before the demo.**
- A free web app expires after about a month unless you press the extend button on the Web tab.
  Deploy a few days before the demo, not weeks before.
- Check the current limits on pythonanywhere.com, because plans change.

## Step 1: Account and upload
1. Create a free "Beginner" account at pythonanywhere.com. Your username becomes part of your link.
2. Open the **Files** tab and upload `aiceo_deploy.zip`.

## Step 2: Unzip and install what is missing
Open **Consoles > Bash** and run:
```
unzip aiceo_deploy.zip
cd aiceo
python3 --version
pip list 2>/dev/null | grep -i -E "flask|pandas|numpy|scikit|joblib|vader"
```
Install only the libraries that are missing from that list (replace NAME):
```
pip install --user NAME
```
VADER is usually missing: `pip install --user vaderSentiment`.
Use the same Python version here and for the web app in Step 4 (for example `python3.12`).

## Step 3: Create the database and train the models (once)
```
python3 setup_demo.py
```
This creates the database and demo accounts, loads the sample tasks, trains Model 2 and Model 3,
and creates the training evidence report. It refuses to run twice, so it cannot erase data by accident.

## Step 4: Create the web app
1. Open the **Web** tab > **Add a new web app** > **Manual configuration** > choose the same Python version.
2. Set **Source code** to `/home/YOURUSERNAME/aiceo`.
3. Click the **WSGI configuration file** link, delete everything, and paste this
   (replace YOURUSERNAME, and generate the secret with `python3 -c "import secrets; print(secrets.token_hex(32))"`):
```python
import os, sys
project = '/home/YOURUSERNAME/aiceo'
if project not in sys.path:
    sys.path.insert(0, project)
os.environ['SECRET_KEY'] = 'PASTE_THE_LONG_RANDOM_TEXT_HERE'
os.environ['COOKIE_SECURE'] = '1'
from app import app as application
```
4. Turn on **Force HTTPS** on the Web tab.
5. Click the green **Reload** button, then open your link.

## Demo accounts
| Role | Email | Password |
|---|---|---|
| CEO | ceo@aiceo.com | ceo123 |
| Manager | manager1@aiceo.com | manager123 |
| Employee | emp001@aiceo.com | employee123 |

## Suggested demo flow (about 6 minutes)
1. Log in as **manager** > Tasks > Employee 001 > **Upload task** > Review it (rating + comment) > Approve.
2. Log out, log in as **employee** (emp001) and show the rating and comment (read-only page).
3. Log in as **CEO** > AI Dashboard: flagged manager M03, under-rated employee E065, click E065 for the AI report.
4. Open **AI Models**: dataset, algorithms, accuracy, saved model files.

## Demo-day tips
- Open the site a few minutes before the teacher does, so the first (slowest) load is already done.
- Do NOT click "Retrain" on the free plan; it uses a lot of CPU. The button works fine on your laptop.
- Charts need internet in the viewer's browser (they load from a CDN).
- Keep a backup plan: on your laptop run `python app.py` and open http://127.0.0.1:5000.

## If something goes wrong
- Web tab > **Error log** shows the reason in plain text.
- `ModuleNotFoundError: No module named X` : run `pip install --user X` and Reload.
- Pages say the model file is missing : run `python3 setup_demo.py` (or `--reset` to start over).
- A scikit-learn version warning : run `python3 setup_demo.py --reset` so the models are trained on the server itself.

## After the demo
- Set `os.environ['SHOW_DEMO_LOGINS'] = '0'` in the WSGI file to hide the demo accounts box.
- Replace the demo passwords before letting real people use the site.
