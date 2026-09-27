.PHONY: install data split-views baselines bilstm distilbert qlora greek greek-eval benchmark serve quantbench lint test docker

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev,serving]"
	$(PY) -m pip install -e ".[llm]" || echo "LLM extras skipped (install on GPU machine)"

## 1) Fetch the raw CSV (size and SHA-256 checked), then build the v2 splits:
##    random (reviews like the training ones), time (later reviews), hotel (unseen hotels)
data:
	$(PY) scripts/fetch_booking_515k.py
	$(PY) -m reviewnlp.data.preprocess --config configs/baselines.yaml
	$(PY) -m reviewnlp.data.preprocess --config configs/baselines_time.yaml
	$(PY) -m reviewnlp.data.preprocess --config configs/baselines_hotel.yaml

## 1b) The dev-selected classical model on each split, with bootstrap 95% intervals
split-views:
	$(PY) scripts/split_views.py

## 2) Classical baselines (TF-IDF + MultinomialNB, TF-IDF + LR-SGD)
baselines:
	$(PY) -m reviewnlp.baselines.classical --config configs/baselines.yaml

## 3) Pure-PyTorch BiLSTM baseline (custom training loop)
bilstm:
	$(PY) -m reviewnlp.baselines.bilstm --config configs/bilstm.yaml

## 4) DistilBERT full fine-tune (mid-size transformer baseline)
distilbert:
	$(PY) -m reviewnlp.llm.train_distilbert --config configs/distilbert.yaml

## 5) Qwen2.5-0.5B-Instruct + QLoRA (run on Colab GPU)
qlora:
	$(PY) -m reviewnlp.llm.train_qlora --config configs/qlora_qwen.yaml

## 5b) GreekBERT on the pinned greek_sa corpus (domain-transfer demo, GPU)
greek:
	$(PY) scripts/train_greek.py --config configs/greek_bert.yaml

## 5c) Score a saved Greek checkpoint on the same configured splits
greek-eval:
	$(PY) scripts/evaluate_greek.py --config configs/greek_bert.yaml --model-dir runs/greek_bert

## 6) Unified benchmark: all models on the identical test set + McNemar tests
benchmark:
	$(PY) -m reviewnlp.evaluation.benchmark --config configs/baselines.yaml

## 7) Serve the best model locally
serve:
	uvicorn reviewnlp.serving.app:app --host 0.0.0.0 --port 8000

## 8) Quantization latency benchmark (dynamic INT8, CPU)
quantbench:
	$(PY) scripts/quantize_distilbert.py --model runs/distilbert/best --dataset data/processed/test.parquet

lint:
	ruff check src tests scripts

test:
	pytest tests -v

docker:
	docker build -f docker/Dockerfile -t hotel-review-nlp:latest .
