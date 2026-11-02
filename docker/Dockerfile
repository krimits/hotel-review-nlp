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

# torch is pinned to the version CI tests against (requirements/ci.lock). Left unpinned, an
# image rebuilt months later serves the model code on a torch nobody tested.
RUN pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu

# Serving stack pinned; the project's own dependencies come with the package below. `--no-deps`
# was wrong: the app imports the ABSA, Greek and analytics routers, whose modules import pandas
# and scikit-learn, so a deps-less install built an image that could not start.
RUN pip install "fastapi==0.143.1" "uvicorn[standard]==0.54.0" \
                   transformers==4.56.2 accelerate==1.10.1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[serving]"

# Not root: the service reads a mounted model and writes nothing but logs.
RUN useradd --create-home --uid 1000 appuser && mkdir -p /models && chown appuser /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "reviewnlp.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
