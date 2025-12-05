from src.data_prep import load_data, split_data
def test_load_and_split():
    df = load_data("data/sample_churn.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.5)
    assert X_train.shape[0] > 0
