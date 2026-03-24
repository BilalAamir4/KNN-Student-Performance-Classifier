import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=["Good", "Average", "Poor"])

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Good", "Avg", "Poor"],
                yticklabels=["Good", "Avg", "Poor"])

    plt.title("Confusion Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.clf()