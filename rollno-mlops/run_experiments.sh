#!/usr/bin/env bash
# Run all 5 MLflow experiments
# Student: Student Name | Roll No: rollno
set -euo pipefail

echo "=== Running 5 MLflow Experiments ==="

# Run 1: v1, LogisticRegression, C=1.0, sepal features
echo "[1/5] v1 | LogisticRegression | C=1.0 | sepal features"
python src/train.py \
  --data-version v1 \
  --model-type logistic \
  --hyperparams '{"C": 1.0}' \
  --features "sepal_length,sepal_width"

# Run 2: v1, LogisticRegression, C=0.1, sepal features
echo "[2/5] v1 | LogisticRegression | C=0.1 | sepal features"
python src/train.py \
  --data-version v1 \
  --model-type logistic \
  --hyperparams '{"C": 0.1}' \
  --features "sepal_length,sepal_width"

# Run 3: v2, LogisticRegression, C=1.0, all 4 features
echo "[3/5] v2 | LogisticRegression | C=1.0 | all 4 features"
python src/train.py \
  --data-version v2 \
  --model-type logistic \
  --hyperparams '{"C": 1.0}' \
  --features "sepal_length,sepal_width,petal_length,petal_width"

# Run 4: v2, LogisticRegression, C=1.0, petal features only
echo "[4/5] v2 | LogisticRegression | C=1.0 | petal features"
python src/train.py \
  --data-version v2 \
  --model-type logistic \
  --hyperparams '{"C": 1.0}' \
  --features "petal_length,petal_width"

# Run 5: v2, RandomForest, n_estimators=100, petal features — BEST MODEL
echo "[5/5] v2 | RandomForest | n_estimators=100 | petal features (best model)"
python src/train.py \
  --data-version v2 \
  --model-type rf \
  --hyperparams '{"n_estimators": 100}' \
  --features "petal_length,petal_width" \
  --save-model models/best_model.pkl

echo "=== All experiments complete. Best model saved to models/best_model.pkl ==="
