import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def basic_feature_engineering(X):
    X = X.copy()
    # convert categorical "Yes"/"No" to 1/0 for some columns
    bool_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for c in bool_cols:
        if c in X.columns:
            X[c] = X[c].map({'Yes':1, 'No':0})
    # Fill missing numeric values
    num_cols = X.select_dtypes(include=['float64','int64']).columns
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    # One-hot encoded categorical columns (small set)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    if len(cat_cols) > 0:
        ohe_arr = ohe.fit_transform(X[cat_cols])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), ohe_df], axis=1)
    return X
