import pickle
from preprocess import load_and_preprocess
from model import KNN
from sklearn.metrics import accuracy_score
from visualize import plot_confusion_matrix

# Load data
(X_train, X_test, y_train, y_test), scaler = load_and_preprocess("../data/student_scores.csv")

# Train model
model = KNN(k=5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")

plot_confusion_matrix(y_test, y_pred)

# Save model
with open("../model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)