import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Create target labels
    def categorize(score):
        if score >= 80:
            return "Good"
        elif score >= 50:
            return "Average"
        else:
            return "Poor"

    df["performance"] = df["exam_score"].apply(categorize)

    X = df[["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]]
    y = df["performance"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler