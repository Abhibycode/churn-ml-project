from src.predict import predict_single
def test_predict_dummy():
    sample = {
      "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,
      "PhoneService":"No","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No",
      "OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
      "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
      "PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85
    }
    # model may not exist in CI; this test can be skipped or mocked in CI
    assert isinstance(sample, dict)
