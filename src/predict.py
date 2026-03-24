import pickle
import numpy as np
import pandas as pd

def predict_new(data):
    with open("../model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    columns = ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]
    data_df = pd.DataFrame([data], columns=columns)

    data_scaled = scaler.transform(data_df)

    prediction = model.predict(data_scaled)
    return prediction[0]


if __name__ == "__main__":
    sample = [5, 7, 80, 75]
    print("Prediction:", predict_new(sample))