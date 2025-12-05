import joblib
import pandas as pd
from features import basic_feature_engineering

def predict_single(model_path, sample_dict):
    df = pd.DataFrame([sample_dict])
    X = basic_feature_engineering(df)
    model = joblib.load(model_path)
    prob = model.predict_proba(X)[:,1]
    return float(prob[0])

if __name__ == "__main__":
    sample = {
      "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,
      "PhoneService":"No","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No",
      "OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
      "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
      "PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85
    }
    print(predict_single("../models/model.joblib", sample))
