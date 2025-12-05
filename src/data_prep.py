import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    # basic cleaning
    df = df.replace(" ", "")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    return df

def split_data(df, target='Churn', test_size=0.2, random_state=42):
    X = df.drop(columns=[target, 'customerID'])
    y = df[target].map({'Yes':1, 'No':0})
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    df = load_data("../data/sample_churn.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    print("Shapes:", X_train.shape, X_test.shape)
