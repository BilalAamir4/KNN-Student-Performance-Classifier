import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_test, y_pred):
    os.makedirs("../results", exist_ok=True)  # ✅

    cm = confusion_matrix(y_test, y_pred, labels=["Good", "Average", "Poor"])

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Good", "Avg", "Poor"],
                yticklabels=["Good", "Avg", "Poor"])

    plt.title("Confusion Matrix")
    plt.tight_layout()  # ✅
    plt.savefig("../results/confusion_matrix.png")
    plt.clf()