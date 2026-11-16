"""
app.py  -  AI CEO Management System (website)
Login (role-based) + Manager dashboard with manual review (rating, comment, approve/reject).
Run:  python app.py   then open  http://127.0.0.1:5000
"""

import csv
import hmac
import io
import logging
import os
import secrets
import sqlite3
import time
from datetime import date, datetime
from functools import wraps

from flask import (Flask, Response, abort, flash, g, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # so aiceo.db and the model files are always found

# Serverless (Vercel) compatibility: copy sqlite db to /tmp for writable filesystem
if os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    import shutil
    tmp_db = "/tmp/aiceo.db"
    src_db = os.path.join(BASE_DIR, "aiceo.db")
    if not os.path.exists(tmp_db) and os.path.exists(src_db):
        shutil.copy2(src_db, tmp_db)
    os.environ["AICEO_DB_PATH"] = tmp_db
    DB_FILE = tmp_db
else:
    DB_FILE = os.environ.get("AICEO_DB_PATH", "aiceo.db")

import ceo_data  # noqa: E402  (must come after chdir and DB_FILE config)
import retrain_models  # noqa: E402
import team_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
MAX_COMMENT = 500
# Show the demo accounts box on the login page? Set SHOW_DEMO_LOGINS=0 to hide it.
SHOW_DEMO_LOGINS = os.environ.get("SHOW_DEMO_LOGINS", "1") == "1"

app = Flask(__name__)
# On a live site set SECRET_KEY in the environment. Without it a random key is used,
# which works but logs everybody out whenever the app restarts.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,   # JavaScript cannot read the login cookie
    SESSION_COOKIE_SAMESITE="Lax",  # extra protection against cross-site requests
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "0") == "1",  # HTTPS only (set 1 online)
)

# ---------- Login attempt limit (stops password guessing) ----------
MAX_FAILS = 5          # wrong passwords allowed ...
LOCK_SECONDS = 300     # ... within this many seconds
_fails = {}            # (ip, email) -> list of timestamps


def _recent_fails(key):
    now = time.time()
    _fails[key] = [t for t in _fails.get(key, []) if now - t < LOCK_SECONDS]
    return _fails[key]

HOME_FOR_ROLE = {"employee": "employee_home", "manager": "manager_home", "ceo": "ceo_home"}


# ---------- Database helpers ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_FILE, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------- CSRF protection (stops other websites from submitting forms for you) ----------
@app.context_processor
def inject_csrf():
    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]
    return {"csrf_token": csrf_token}


def check_csrf():
    sent = request.form.get("csrf_token", "")
    if not sent or not hmac.compare_digest(sent, session.get("csrf", "")):
        abort(400)


# ---------- Access control ----------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "info")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def role_required(*roles):
    """Only allow the listed roles. Everyone else gets 'Access denied'."""
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if session.get("role") not in roles:
                return render_template("denied.html"), 403
            return view(*args, **kwargs)
        return wrapped
    return decorator


# ---------- Friendly error pages ----------
@app.errorhandler(400)
def bad_request(_e):
    return render_template("error.html", code=400, title="Request not accepted",
                           message="Something was wrong with that request. Please go back and try again."), 400


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, title="Page not found",
                           message="We could not find what you were looking for."), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Server error: %s", e)
    return render_template("error.html", code=500, title="Something went wrong",
                           message="An error happened on our side. Please try again in a moment."), 500


# ---------- Login / logout ----------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for(HOME_FOR_ROLE[session["role"]]))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        key = (request.remote_addr, email)
        if len(_recent_fails(key)) >= MAX_FAILS:
            flash("Too many failed attempts. Please wait a few minutes and try again.", "error")
            return render_template("login.html", email=email, show_demo=SHOW_DEMO_LOGINS), 429

        user = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            _fails.setdefault(key, []).append(time.time())
            flash("Invalid email or password.", "error")
            return render_template("login.html", email=email, show_demo=SHOW_DEMO_LOGINS), 401
        _fails.pop(key, None)

        session.clear()
        session["user_id"] = user["id"]
        session["name"] = user["name"]
        session["role"] = user["role"]
        session["employee_id"] = user["employee_id"]
        session["manager_id"] = user["manager_id"]
        session["csrf"] = secrets.token_hex(16)
        return redirect(url_for(HOME_FOR_ROLE[user["role"]]))

    return render_template("login.html", email="", show_demo=SHOW_DEMO_LOGINS)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------- Manager dashboard ----------
@app.route("/manager")
@role_required("manager")
def manager_home():
    mid = session["manager_id"]
    return render_template("manager_home.html", active="home",
                           team=team_data.team_stats(mid),
                           n_employees=len(team_data.team_employee_ids(mid)))


@app.route("/manager/tasks")
@role_required("manager")
def manager_tasks():
    return render_template("manager_tasks.html", active="tasks",
                           employees=team_data.team_overview(session["manager_id"]))


@app.route("/manager/tasks/<emp_id>")
@role_required("manager")
def manager_employee(emp_id):
    mid = session["manager_id"]
    # A manager can only open employees from their own team
    if emp_id not in team_data.team_employee_ids(mid):
        abort(404)
    overview = {e["id"]: e for e in team_data.team_overview(mid)}
    start_filter = request.args.get("f", "all")
    if start_filter not in ("all", "approved", "pending", "rejected"):
        start_filter = "all"
    return render_template("manager_employee.html", active="tasks", emp=overview[emp_id],
                           tasks=team_data.employee_tasks(mid, emp_id),
                           start_filter=start_filter, today=date.today().isoformat())


@app.route("/manager/tasks/<emp_id>/new", methods=["POST"])
@role_required("manager")
def manager_new_task(emp_id):
    """Manager uploads a task record for one of their employees. It starts as pending."""
    check_csrf()
    mid = session["manager_id"]
    if emp_id not in team_data.team_employee_ids(mid):
        abort(404)
    back = url_for("manager_employee", emp_id=emp_id, f="pending")

    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()[:500]
    try:
        deadline = date.fromisoformat(request.form.get("deadline", ""))
        completed = date.fromisoformat(request.form.get("completed_date", ""))
    except ValueError:
        flash("Please enter valid dates.", "error")
        return redirect(back)

    if not 1 <= len(title) <= 100:
        flash("Task title is required (up to 100 characters).", "error")
        return redirect(back)
    low, high = date(2020, 1, 1), date(2100, 1, 1)
    if not (low <= deadline <= high and low <= completed <= high):
        flash("Dates must be between 2020 and 2100.", "error")
        return redirect(back)
    if completed > date.today():
        flash("The completion date cannot be in the future.", "error")
        return redirect(back)

    db = get_db()
    for _ in range(3):  # retry if two uploads happen at the exact same moment
        n = db.execute("SELECT IFNULL(MAX(CAST(SUBSTR(task_id, 2) AS INTEGER)), 0) + 1 FROM tasks").fetchone()[0]
        task_id = f"T{n:05d}"
        try:
            db.execute(
                "INSERT INTO tasks (task_id, employee_id, manager_id, title, description, deadline, "
                "completed_date, status, rating, comment, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, '', NULL)",
                (task_id, emp_id, mid, title, description, deadline.isoformat(), completed.isoformat()))
            db.commit()
            flash(f"Task {task_id} uploaded for {team_data.employee_name(emp_id)}. It is pending your review.", "success")
            return redirect(back)
        except sqlite3.IntegrityError:
            db.rollback()
    flash("Could not save the task. Please try again.", "error")
    return redirect(back)


@app.route("/manager/tasks/<task_id>/delete", methods=["POST"])
@role_required("manager")
def manager_delete_task(task_id):
    """Remove a task uploaded by mistake. Only possible while it is still pending."""
    check_csrf()
    db = get_db()
    task = db.execute("SELECT employee_id FROM tasks WHERE task_id = ? AND manager_id = ? AND status = 'pending'",
                      (task_id, session["manager_id"])).fetchone()
    if task is None:
        abort(404)
    db.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
    db.commit()
    flash(f"Task {task_id} was deleted.", "info")
    return redirect(url_for("manager_employee", emp_id=task["employee_id"], f=request.form.get("f", "all")))


@app.route("/manager/review/<task_id>", methods=["POST"])
@role_required("manager")
def review_task(task_id):
    """Manager saves a rating, a comment and a decision for one task."""
    check_csrf()
    db = get_db()
    # The task must belong to THIS manager's team
    task = db.execute("SELECT * FROM tasks WHERE task_id = ? AND manager_id = ?",
                      (task_id, session["manager_id"])).fetchone()
    if task is None:
        abort(404)

    back = url_for("manager_employee", emp_id=task["employee_id"],
                   f=request.form.get("f", "all"))

    status = request.form.get("status", "")
    if status not in ("approved", "rejected", "pending"):
        abort(400)

    rating = None
    raw = request.form.get("rating", "").strip()
    if raw:
        if not raw.isdigit() or not 1 <= int(raw) <= 5:
            flash("Rating must be a number from 1 to 5.", "error")
            return redirect(back)
        rating = int(raw)
    if status in ("approved", "rejected") and rating is None:
        flash("Please choose a rating before approving or rejecting.", "error")
        return redirect(back)
    if status == "pending":
        rating = None  # back to pending means not rated yet

    comment = request.form.get("comment", "").strip()[:MAX_COMMENT]

    db.execute(
        "UPDATE tasks SET status = ?, rating = ?, comment = ?, reviewed_at = ? WHERE task_id = ?",
        (status, rating, comment, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task_id),
    )
    db.commit()
    flash(f"Saved: {task_id} is now {status}.", "success")
    return redirect(back)


@app.route("/manager/insights")
@role_required("manager")
def manager_insights():
    return render_template("manager_insights.html", active="insights",
                           d=team_data.insights(session["manager_id"]))


@app.route("/manager/reports")
@role_required("manager")
def manager_reports():
    return render_template("manager_reports.html", active="reports",
                           r=team_data.reports(session["manager_id"]))


def _csv_response(filename, header, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.route("/manager/reports/completion.csv")
@role_required("manager")
def download_completion():
    rows = [[e["id"], e["name"], e["total"], e["approved"], e["pending"], e["rejected"],
             e["completion_pct"], e["on_time_pct"], e["avg_rating"], e["score"]]
            for e in team_data.team_overview(session["manager_id"])]
    return _csv_response("task_completion_report.csv",
                         ["Employee ID", "Name", "Total", "Approved", "Pending", "Rejected",
                          "Completion %", "On-time %", "Avg rating", "Score"], rows)


@app.route("/manager/reports/monthly.csv")
@role_required("manager")
def download_monthly():
    rows = [[m["month"], m["total"], m["approved"], m["pending"], m["rejected"],
             m["on_time_pct"], m["avg_rating"], m["score"]]
            for m in team_data.reports(session["manager_id"])["monthly"]]
    return _csv_response("monthly_summary.csv",
                         ["Month", "Tasks", "Approved", "Pending", "Rejected",
                          "On-time %", "Avg rating", "Score"], rows)


# ---------- Employee dashboard (read-only: tasks are uploaded by the manager) ----------
@app.route("/employee")
@role_required("employee")
def employee_home():
    emp_id, mid = session["employee_id"], session["manager_id"]
    start_filter = request.args.get("f", "all")
    if start_filter not in ("all", "approved", "pending", "rejected"):
        start_filter = "all"
    return render_template("employee_dashboard.html", active="employee",
                           emp=team_data.employee_stats(mid, emp_id),
                           tasks=team_data.employee_tasks(mid, emp_id),
                           start_filter=start_filter)


@app.route("/ceo")
@role_required("ceo")
def ceo_home():
    return render_template("ceo_dashboard.html", active="dashboard", d=ceo_data.analyze())


@app.route("/ceo/rerun", methods=["POST"])
@role_required("ceo")
def ceo_rerun():
    check_csrf()
    ceo_data.analyze(force=True)
    flash("AI analysis re-run on the latest saved reviews.", "success")
    return redirect(url_for("ceo_home"))


@app.route("/ceo/employee/<emp_id>")
@role_required("ceo")
def ceo_employee(emp_id):
    rep = ceo_data.employee_report(emp_id)
    if rep is None:
        abort(404)
    return render_template("ceo_employee.html", active="dashboard", r=rep)


@app.route("/ceo/retrain", methods=["POST"])
@role_required("ceo")
def ceo_retrain():
    check_csrf()
    try:
        r = retrain_models.run()
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("ceo_models"))
    ceo_data.analyze(force=True)
    flash(f"Models retrained on {r['reviewed_rows']} reviewed tasks from the database "
          f"(average error {r['mae']} stars). Old models were backed up.", "success")
    return redirect(url_for("ceo_models"))


@app.route("/ceo/models")
@role_required("ceo")
def ceo_models():
    return render_template("ceo_models.html", active="models", m=ceo_data.model_report(),
                           log=ceo_data.training_log())


if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print("Database not found. Run  python init_db.py  then  python load_tasks.py  first.")
    else:
        if not os.environ.get("SECRET_KEY"):
            app.secret_key = "dev-only-change-this-secret-key"  # stable key for local development
        port = int(os.environ.get("PORT", 5000))
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
        except OSError:
            if "PORT" not in os.environ:
                port = 5001
        app.run(debug=True, port=port)
