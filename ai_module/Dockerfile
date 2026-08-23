# ==============================
# Stage 1 — Base
# ==============================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Stage 2 — Dependencies
# ==============================
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ==============================
# Stage 3 — Production
# ==============================
FROM dependencies AS production

COPY . .

RUN mkdir -p logs models/production models/checkpoints data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.service.ai_service:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]