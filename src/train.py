import mlflow
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from data_prep import load_data, split_data
from features import basic_feature_engineering

MLFLOW_EXP = "churn-experiment"

def train(config):
    df = load_data(config['data_path'])
    X_train, X_test, y_train, y_test = split_data(df)
    X_train = basic_feature_engineering(X_train)
    X_test = basic_feature_engineering(X_test)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=config.get('n_estimators',100), random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, preds)
        acc = accuracy_score(y_test, (preds>0.5).astype(int))

        mlflow.log_param("n_estimators", config.get('n_estimators',100))
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        # save model
        os.makedirs(config['model_dir'], exist_ok=True)
        model_path = os.path.join(config['model_dir'], "model.joblib")
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"Trained model AUC={auc:.4f} ACC={acc:.4f}")
        return model_path

if __name__ == "__main__":
    cfg = {"data_path":"../data/sample_churn.csv","model_dir":"../models","n_estimators":100}
    train(cfg)
