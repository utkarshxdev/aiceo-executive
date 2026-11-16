# AI CEO Management System - how to start

Folder location:  C:\AICEO2

## First time only (about a minute)
    cd C:\AICEO2
    pip install -r requirements.txt
    python setup_demo.py

setup_demo.py creates the database + demo accounts, loads the sample tasks,
trains Model 2 and Model 3, and creates the training evidence report.
(It refuses to run if aiceo.db already exists. Use --reset to start over.)

## Every time you want to run the website
    cd C:\AICEO2
    python app.py
Then open  http://127.0.0.1:5000

## Demo accounts
    CEO       ceo@aiceo.com        ceo123
    Manager   manager1@aiceo.com   manager123      (manager1 ... manager10)
    Employee  emp001@aiceo.com     employee123     (emp001 ... emp100)

## What is where
    app.py                    website (login, manager, employee, CEO pages)
    team_data.py              numbers for manager/employee pages (reads the database)
    ceo_data.py               runs the trained AI models for the CEO pages
    init_db.py, load_tasks.py database + sample tasks
    generate_data.py          creates tasks_data.csv (the 10,000-task training dataset)
    train_model2.py           trains Model 2 (Random Forest)  -> rating_model.pkl
    train_model3.py           trains Model 3 (Isolation Forest) -> anomaly_model.pkl
    model1_score.py           Model 1 (score formula)
    model4_sentiment.py       Model 4 (comment sentiment)
    model5_summary.py         Model 5 (report writer)
    evaluate_models.py        training evidence -> model_report.json
    retrain_models.py         retrain on database history
    generate_all_reports.py   writes a report for every employee (all_reports.txt)
    DEPLOY_GUIDE.md           how to put it online (PythonAnywhere)
    templates/, static/       the pages and the design
