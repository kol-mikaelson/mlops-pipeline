FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/best_model.pkl

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
