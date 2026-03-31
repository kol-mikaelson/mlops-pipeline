# MLOps Pipeline Assignment Report
**Student Name:** Student Name
**Roll No:** rollno
**Docker Hub:** dockerhubuser/rollno-mlops

---

## 1. MLflow Experiment Results

> Fill in after running `run_experiments.sh`

| Run | Data | Model | Hyperparams | Features | Accuracy | F1 Score |
|-----|------|-------|-------------|----------|----------|----------|
| 1 | v1 (100 rows) | LogisticRegression | C=1.0 | sepal_length, sepal_width | ___ | ___ |
| 2 | v1 (100 rows) | LogisticRegression | C=0.1 | sepal_length, sepal_width | ___ | ___ |
| 3 | v2 (150 rows) | LogisticRegression | C=1.0 | all 4 features | ___ | ___ |
| 4 | v2 (150 rows) | LogisticRegression | C=1.0 | petal_length, petal_width | ___ | ___ |
| 5 | v2 (150 rows) | RandomForest | n_estimators=100 | petal_length, petal_width | ___ | ___ |

---

## 2. Analysis Questions

**Q1. Which model performed best and why?**
Run 5 (RandomForest, v2, petal features) is expected to perform best. Random forests are ensemble methods that average multiple decision trees, reducing variance and handling non-linear boundaries better than logistic regression on this dataset.

**Q2. How does data version (v1 vs v2) affect performance?**
v2 includes all 150 rows vs 100 in v1, giving the model more training examples. The Iris dataset is balanced across 3 classes; v1 truncates the virginica class, making multi-class metrics worse.

**Q3. How does feature selection impact accuracy?**
Petal features (petal_length, petal_width) are more discriminative for Iris classification than sepal features. Runs using petal features are expected to score higher than runs using only sepal features.

**Q4. What is the effect of regularization strength C on LogisticRegression?**
Lower C (Run 2, C=0.1) applies stronger L2 regularization, potentially underfitting when data is limited. Higher C (C=1.0) allows the model more freedom to fit the training data, generally better for this small dataset.

**Q5. Why use MLflow for experiment tracking?**
MLflow provides reproducible experiment logging (params, metrics, artifacts) and a UI to compare runs, making it easy to identify the best model without manually tracking results in spreadsheets.

**Q6. What is the role of DVC in this pipeline?**
DVC versions large data files and ML artifacts outside git (in S3), tracks pipeline stages, and ensures reproducibility via `dvc repro`. It bridges the gap between code versioning (git) and data/model versioning.

**Q7. Why containerize the API with Docker?**
Docker ensures the FastAPI app and its dependencies run identically in development, CI, and production environments, eliminating "works on my machine" issues.

**Q8. What does the GitHub Actions workflow automate?**
It automates: dependency installation, data retrieval via DVC, all 5 MLflow training runs, Docker image build, and push to Docker Hub — triggered on every push to main.

**Q9. How is the best model selected and served?**
Run 5 is hardcoded as the best model based on expected performance. Its artifact is saved to `models/best_model.pkl` and loaded at FastAPI startup. In a production system, MLflow's model registry would be used to promote the best run dynamically.

**Q10. What security practices were followed?**
AWS credentials and Docker Hub tokens are stored as GitHub Secrets and never hardcoded in any file. MLflow tracking URI is configurable via environment variable. Input validation is enforced via Pydantic in the API.

---

## 3. Screenshots Required

- [ ] MLflow UI showing all 5 runs in `rollno_experiment`
- [ ] MLflow run detail page for best run (Run 5)
- [ ] `GET /health` response in terminal or browser
- [ ] `POST /predict` response with valid input
- [ ] GitHub Actions workflow run (green)
- [ ] Docker Hub repository page showing `rollno-mlops:latest`
- [ ] DVC DAG (`dvc dag` output)

---

## 4. Links

| Resource | URL |
|----------|-----|
| GitHub Repo | `https://github.com/<username>/rollno-mlops` |
| Docker Hub Image | `https://hub.docker.com/r/dockerhubuser/rollno-mlops` |
| MLflow UI (local) | `http://localhost:5000` |
| FastAPI Docs | `http://localhost:8000/docs` |
