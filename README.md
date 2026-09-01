# Explainable Risk-Ranked Predictive Maintenance

Predicts *if* a machine will fail, *why* it's failing, and *which one to fix first* — not just another binary alert system.

## The problem

Most predictive maintenance tools tell you a machine is *going* to fail — and stop there. That leaves technicians to figure out:
- **What** actually broke (tool wear? overheating? power issue?)
- **Which** machine to fix first when five alarms go off at once



## What this does

Upload a file with machine sensor readings, and the app runs it through a two-stage ML pipeline that:

1. **Predicts which machines are at risk of failure.**
2. **Identifies the specific type of failure.**
3. **Ranks machines by maintenance priority, helping teams decide what to fix first when time and technicians are limited.**




## Key features

- **Two-stage cascade pipeline** — a fast binary check, followed by detailed fault classification only on the machines that need it
- **Failure-type diagnosis** — flags specific fault types (Tool Wear Failure, Overstrain Failure, Heat Dissipation Failure, Power Failure) instead of a generic "at risk" label
- **Priority ranking** — a weighted risk score sorts machines from most to least urgent, with confidence and status (Critical / Monitor / Stable)
- **SHAP explainability** — every prediction comes with the sensor readings that actually drove it, so nothing is a black box
- **Web dashboard + downloadable CSV** — deployed as a Flask app with a browsable priority queue and a one-click CSV export

## How it works

```
Sensor data upload
        │
        ▼
Preprocessing (data_prep.py)
        │
        ▼
Stage 1 — XGBoost binary classifier
   "Is this machine at risk?"
        │
   ┌────┴────┐
   │         │
 Healthy   Flagged
   │         │
   │         ▼
   │   Stage 2 — Logistic Regression (multi-output)
   │   "What type of failure is it?"
   │   (Tool Wear / Heat Dissipation / Power Failure / Overstrain)
   │         │
   │         ▼
   │   SHAP TreeExplainer
   │   "Which sensor readings caused this?"
   │         │
   └────┬────┘
        ▼
Priority score = 0.4·(Stage 1 risk) + 0.3·(Stage 2 confidence) + 0.3·(SHAP magnitude)
   (MinMax-normalized per batch)
        │
        ▼
Ranked priority queue → Flask dashboard / CSV export
```

Stage 1 uses `scale_pos_weight` to handle the class imbalance in the data — only about 3.4% of machines actually fail, so without correcting for that the model would just predict "healthy" for everything.

## Dataset

- **[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** — the primary dataset, with sensor/operational readings including Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear, plus labeled failure types.
- **NASA C-MAPSS FD001** — used separately to test cross-dataset generalization, mapping Remaining Useful Life (RUL) into urgency tiers.

## Results

| Metric | Value |
|---|---|
| Stage 1 Precision | 0.776 |
| Stage 1 Recall | 0.765 |
| Stage 2 Isolated Micro-F1 | 0.920 |
| Stage 2 Isolated Exact Match | 0.853 |
| Cascade Contamination Rate | 0.224 |
| **True Cascade Exact Match** | **0.761** |



Tested Logistic Regression, Decision Trees, and Random Forest for Stage 1 before settling on XGBoost for the binary detection step.

## Tech stack

- **XGBoost** — Stage 1 binary failure detection
- **Scikit-learn** — `MultiOutputClassifier` wrapping Logistic Regression for Stage 2, plus preprocessing and evaluation metrics
- **SHAP** — `TreeExplainer` for feature-level explanations behind each prediction
- **Pandas / NumPy / SciPy** — data handling and numerical computation
- **Flask** — REST API + web dashboard
- **Gunicorn** — production WSGI server
- **Joblib** — model serialization (no retraining needed at inference time)
- **Render** — deployment

## Repository structure

```
├── data/
│   ├── processed/
│   │   ├── ai4i2020.csv
│   │   └── sample_new_readings.csv
├── models/
│   ├── pipeline_metadata.joblib
│   ├── shap_background.joblib
│   ├── stage1_scaler.joblib
│   ├── stage1_xgb_model.joblib
│   └── stage2_xgb_model.joblib
├── notebook/
│   └── eda.py
├── results/
│   ├── eda_plots/
│   ├── stage1/
│   ├── stage2/
│   ├── cascade_evaluation.csv
│   ├── new_ranking.csv
│   └── priority_ranking.csv
├── src/
│   ├── stage1/
│   ├── stage2/
│   ├── cascade_pipeline.py
│   ├── data_prep.py
│   └── predict_batch.py
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
├── app.py
└── requirement.txt
```

## Getting started

The app is live at the demo link above, but here's how to run it locally:

```bash
# Clone the repo
git clone https://github.com/vanshika15manchanda/Explainable-Risk-Ranked-Predictive-Maintenance.git
cd Explainable-Risk-Ranked-Predictive-Maintenance

# Set up a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Run the app
python app.py
```

For a production-style run:

```bash
gunicorn app:app
```

Once it's running, open the dashboard in your browser, upload a sensor readings file, and you'll get back the priority queue you see in the screenshot below.

## Dashboard

The priority queue view shows every machine sorted by risk, with its predicted failure type, risk score, and confidence — filterable by status, failure type, and risk level, with a one-click CSV export.

<img width="1600" height="725" alt="WhatsApp Image 2026-09-02 at 00 37 25" src="https://github.com/user-attachments/assets/ea0d72b7-c7e9-43af-8bde-98a25c77b3ac" />
<img width="1557" height="372" alt="image" src="https://github.com/user-attachments/assets/d5a6eb87-f41f-493c-bc42-2542885d7239" />
<img width="1591" height="302" alt="image" src="https://github.com/user-attachments/assets/2e2ac7a2-d5b7-40ac-8adc-1ec0cda4a13b" />


Check out the live web application here:https://docs.google.com/presentation/d/1GIkwUBTm_aT4GKF5ULRTLO8FJxJUbOdb/edit?usp=sharing&ouid=117704189816769727589&rtpof=true&sd=true



## Team

Built by — [@vanshika15manchanda](https://github.com/vanshika15manchanda), [@vijayajaiswal398](https://github.com/vijayajaiswal398), [@ishikasingh2906](https://github.com/ishikasingh2906)
