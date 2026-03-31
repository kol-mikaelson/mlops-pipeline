"""
Training script for MLOps pipeline.
Student: Purandhar Reddy | Roll No: 2022bcs0179
"""

import argparse
import json
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

STUDENT_NAME = "Purandhar Reddy"
ROLL_NO = "2022bcs0179"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(data_version: str, features: list[str]) -> tuple:
    path = os.path.join(DATA_DIR, f"{data_version}_data.csv")
    df = pd.read_csv(path)
    X = df[features]
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(model_type: str, hyperparams: dict):
    if model_type == "logistic":
        return LogisticRegression(**hyperparams, max_iter=1000, random_state=42)
    if model_type == "rf":
        return RandomForestClassifier(**hyperparams, random_state=42)
    if model_type == "svm":
        return SVC(**hyperparams, random_state=42)
    raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True, choices=["v1", "v2"])
    parser.add_argument("--model-type", required=True, choices=["logistic", "rf", "svm"])
    parser.add_argument("--hyperparams", default="{}", type=str)
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--save-model", default=None, type=str,
                        help="If set, save model pickle to this path")
    args = parser.parse_args()

    hyperparams = json.loads(args.hyperparams)
    features = [f.strip() for f in args.features.split(",")]

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{ROLL_NO}_experiment")

    X_train, X_test, y_train, y_test = load_data(args.data_version, features)
    model = build_model(args.model_type, hyperparams)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("features_used", args.features)
        mlflow.log_params(hyperparams)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"accuracy={acc:.4f}  f1={f1:.4f}")

        metrics = {
            "name": STUDENT_NAME,
            "roll_no": ROLL_NO,
            "model_type": args.model_type,
            "data_version": args.data_version,
            "features": features,
            "hyperparams": hyperparams,
            "accuracy": acc,
            "f1_score": f1,
        }
        with open("metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
            with open(args.save_model, "wb") as fh:
                pickle.dump(model, fh)
            mlflow.log_artifact(args.save_model)
            print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
