#!/usr/bin/env bash
# API validation script
# Student: Student Name | Roll No: rollno
set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"

echo "=== Testing API at $BASE_URL ==="

echo ""
echo "[1/2] GET /health"
curl -sf "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "[2/2] POST /predict (petal_length=1.4, petal_width=0.2 — expected: setosa)"
curl -sf "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"petal_length": 1.4, "petal_width": 0.2}' | python3 -m json.tool

echo ""
echo "[bonus] POST /predict (petal_length=5.1, petal_width=1.8 — expected: virginica)"
curl -sf "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"petal_length": 5.1, "petal_width": 1.8}' | python3 -m json.tool

echo ""
echo "=== All tests passed ==="
