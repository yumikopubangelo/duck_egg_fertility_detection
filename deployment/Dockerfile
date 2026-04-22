# Multi-stage build for duck egg fertility detection app
FROM python:3.11-slim-bookworm as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-web.txt deployment/requirements-deploy.txt ./

# Install production dependencies first
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/
COPY web/ ./web/
COPY configs/ ./configs/
COPY models/ ./models/
COPY data/ ./data/

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set Flask environment variables
ENV FLASK_APP=web.api.app:app
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the application using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "web.api.app:app"]