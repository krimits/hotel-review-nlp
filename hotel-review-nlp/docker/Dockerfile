# CPU inference image for the sentiment API.
# Build:  docker build -f docker/Dockerfile -t hotel-review-nlp .
# Run:    docker run -p 8000:8000 -e MODEL_TYPE=stub hotel-review-nlp
#         (mount runs/ and set MODEL_TYPE=encoder MODEL_PATH=/models/distilbert for the real model)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_TYPE=stub \
    MODEL_PATH=""

WORKDIR /app

# CPU-only torch first: slimmer than the CUDA default wheel
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[serving]" && pip install transformers accelerate

# models are mounted at runtime, never baked into the image
RUN mkdir -p /models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "reviewnlp.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
