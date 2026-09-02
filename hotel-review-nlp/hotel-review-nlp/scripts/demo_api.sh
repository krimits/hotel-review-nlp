#!/usr/bin/env bash
# End-to-end local demo: start the API with the trained encoder and hit it.
# Usage: bash scripts/demo_api.sh [MODEL_PATH]
set -euo pipefail

MODEL_PATH="${1:-runs/distilbert}"

echo "== starting API with MODEL_TYPE=encoder, MODEL_PATH=$MODEL_PATH =="
MODEL_TYPE=encoder MODEL_PATH="$MODEL_PATH" uvicorn reviewnlp.serving.app:app --port 8000 &
API_PID=$!
sleep 6   # let the model load

trap "kill $API_PID 2>/dev/null || true" EXIT

echo; echo "== /health ==";               curl -s http://127.0.0.1:8000/health | python3 -m json.tool
echo; echo "== /predict (positive) ==";   curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Perfect stay, amazing breakfast and the staff treated us like royalty."}' | python3 -m json.tool
echo; echo "== /predict (negative) ==";   curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Room smelled like smoke, the AC was broken and reception ignored us."}' | python3 -m json.tool
echo; echo "== /predict/batch ==";        curl -s -X POST http://127.0.0.1:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["Loved every minute of our stay.", "Worst hotel experience of my life."]}' | python3 -m json.tool
