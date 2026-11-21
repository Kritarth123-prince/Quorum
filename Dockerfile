FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY retrieval_pipeline/requirements.txt /tmp/requirements.txt
# CPU-only torch wheel keeps the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m playwright install --with-deps chromium

COPY pyproject.toml ./
COPY retrieval_pipeline ./retrieval_pipeline

EXPOSE 8000
CMD ["uvicorn", "retrieval_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
