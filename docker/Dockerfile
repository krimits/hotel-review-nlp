# CPU inference image for the sentiment API.
# Build:  docker build -f docker/Dockerfile -t hotel-review-nlp .
# Run:    docker run -p 8000:8000 -e MODEL_TYPE=stub hotel-review-nlp
#         Real weights are mounted, never baked in:
#         docker run -p 8000:8000 -v "$(pwd)/models:/models:ro" \
#             -e MODEL_TYPE=encoder -e MODEL_PATH=/models/distilbert hotel-review-nlp

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
# Pinned to the versions CI tests against, so an image rebuilt months from now
# serves the same model code that produced the published numbers.
RUN pip install -e ".[serving]" && pip install transformers==4.56.2 accelerate==1.10.1

# models are mounted at runtime, never baked into the image
RUN mkdir -p /models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "reviewnlp.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
