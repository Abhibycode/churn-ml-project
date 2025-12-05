# Churn Prediction — End-to-End ML + MLOps Demo

## Summary
This repo demonstrates an end-to-end churn prediction pipeline: ingestion, EDA, feature engineering, model training with MLflow, and a FastAPI service for serving the model.

## Quickstart (local)

1. Clone:
```bash
git clone https://github.com/your-username/churn-ml-project.git
cd churn-ml-project
Create venv & install:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Train model (this logs to mlflow):

bash
Copy code
python src/train.py
# or run via python -c "from src.train import train; train({...})"
Start API (after training ensure models/model.joblib exists):

bash
Copy code
cd api
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
Predict:

bash
Copy code
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" \
-d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
MLflow
Start UI:

bash
Copy code
mlflow ui --port 5000
Docker
Build & run:

bash
Copy code
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api
Tests
bash
Copy code
pytest -q
Contact
Abhishek Kongari — abhishekongari@gmail.com


---

# ✅ Commit & release suggestions
- Commit message: `feat: add churn prediction E2E pipeline (data, train, api, mlflow)`
- Tag: `v0.1.0` — Initial E2E demo

---
