"""
Data preprocessing script for MLOps pipeline.
Student: Student Name | Roll No: rollno
"""

import os
import pandas as pd
from sklearn.datasets import load_iris

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_and_save_raw():
    os.makedirs(RAW_DIR, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    raw_path = os.path.join(RAW_DIR, "iris.csv")
    df.to_csv(raw_path, index=False)
    return df


def create_v1(df: pd.DataFrame) -> pd.DataFrame:
    """First 100 rows, only sepal features."""
    return df[["sepal_length", "sepal_width", "target"]].iloc[:100].copy()


def create_v2(df: pd.DataFrame) -> pd.DataFrame:
    """All 150 rows, all 4 features."""
    return df.copy()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_and_save_raw()

    v1 = create_v1(df)
    v1.to_csv(os.path.join(DATA_DIR, "v1_data.csv"), index=False)
    print(f"v1_data.csv saved: {len(v1)} rows, {list(v1.columns)}")

    v2 = create_v2(df)
    v2.to_csv(os.path.join(DATA_DIR, "v2_data.csv"), index=False)
    print(f"v2_data.csv saved: {len(v2)} rows, {list(v2.columns)}")


if __name__ == "__main__":
    main()
