# Stage 1: Build frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend + serve frontend
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies (OpenCV + Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml .python-version ./
RUN uv sync --no-dev

# Copy application code
COPY *.py ./
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create directories
RUN mkdir -p input output

# Expose port
EXPOSE 8000

# Run backend (serves API + static frontend)
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
