import joblib
import pandas as pd
from sklearn.metrics import classification_report
from data_prep import load_data, split_data
from features import basic_feature_engineering

def evaluate(model_path):
    df = load_data("../data/sample_churn.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    X_test = basic_feature_engineering(X_test)
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate("../models/model.joblib")
