.PHONY: install data baselines bilstm bilstm-threshold distilbert qlora benchmark serve lint test docker

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev,serving,tracking]"
	$(PY) -m pip install -e ".[llm]" || echo "LLM extras skipped (install on GPU machine)"

## 1) Build the processed dataset from the raw Kaggle CSV (run once)
data:
	$(PY) -m reviewnlp.data.preprocess --config configs/baselines.yaml

## 2) Classical baselines (TF-IDF + MultinomialNB, TF-IDF + LR-SGD)
baselines:
	$(PY) -m reviewnlp.baselines.classical --config configs/baselines.yaml

## 3) Pure-PyTorch BiLSTM baseline (custom training loop)
bilstm:
	$(PY) -m reviewnlp.baselines.bilstm --config configs/bilstm.yaml

## 3b) Seed-100 BiLSTM with threshold selected exclusively on the dev set
bilstm-threshold:
	$(PY) -m reviewnlp.baselines.bilstm --config configs/bilstm_seed100_threshold.yaml

## 4) DistilBERT full fine-tune (mid-size transformer baseline)
distilbert:
	$(PY) -m reviewnlp.llm.train_distilbert --config configs/distilbert.yaml

## 5) Qwen2.5-0.5B-Instruct + QLoRA (run on Colab GPU)
qlora:
	$(PY) -m reviewnlp.llm.train_qlora --config configs/qlora_qwen.yaml

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
