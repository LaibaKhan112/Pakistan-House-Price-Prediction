import joblib
def predict(data):
    clf = joblib.load("house_price_predictor.sav")
    return clf.predict(data)