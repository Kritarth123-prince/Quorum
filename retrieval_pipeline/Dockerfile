FROM python:3.11-slim

WORKDIR /app

# System dependencies for Playwright + lxml
RUN apt-get update && apt-get install -y \
    curl wget gnupg \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium

COPY . .

EXPOSE 8001

CMD ["uvicorn", "retrieval_pipeline.api:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--log-level", "info"]
