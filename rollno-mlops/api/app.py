"""
FastAPI inference service for MLOps pipeline.
Student: Student Name | Roll No: rollno
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STUDENT_NAME = "Student Name"
ROLL_NO = "rollno"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"))

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run run_experiments.sh first.")
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    print(f"Model loaded from {MODEL_PATH}")
    yield


app = FastAPI(title="Iris Classifier API", lifespan=lifespan)


class PredictRequest(BaseModel):
    petal_length: float = Field(..., description="Petal length in cm", gt=0)
    petal_width: float = Field(..., description="Petal width in cm", gt=0)
    sepal_length: Optional[float] = Field(None, description="Sepal length in cm (optional)", gt=0)
    sepal_width: Optional[float] = Field(None, description="Sepal width in cm (optional)", gt=0)


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    name: str
    roll_no: str


CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.get("/health")
def health():
    return {"name": STUDENT_NAME, "roll_no": ROLL_NO, "status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Best model (Run 5) uses petal_length, petal_width
    features = np.array([[req.petal_length, req.petal_width]])
    pred = int(model.predict(features)[0])
    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES.get(pred, "unknown"),
        name=STUDENT_NAME,
        roll_no=ROLL_NO,
    )
