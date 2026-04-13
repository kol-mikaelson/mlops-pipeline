"""
Training script for MLOps pipeline.
Student: Purandhar Reddy | Roll No: 2022bcs0179
"""

import json
import os
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

STUDENT_NAME = "Purandhar Reddy"
ROLL_NO = "2022bcs0179"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    iris = load_iris()
    X = iris.data[:, 2:4]  # petal_length, petal_width
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"accuracy={acc:.4f}  f1={f1:.4f}")

    metrics = {
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO,
        "model_type": "rf",
        "features": ["petal_length", "petal_width"],
        "accuracy": acc,
        "f1_score": f1,
    }
    with open("metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
