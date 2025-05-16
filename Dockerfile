# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre \
    netcat-traditional \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and logs
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models /app/data/logs \
    && chown -R appuser:appuser /app/data

# Copy project files
COPY . .

# Create entrypoint scripts for different components
RUN mkdir -p /app/scripts \
    && echo '#!/bin/bash\necho "Starting data collector..."\npython -m app.data_collector' > /app/scripts/start-collector.sh \
    && echo '#!/bin/bash\necho "Starting model trainer..."\npython -c "from app.ml_model import StockPricePredictor; predictor = StockPricePredictor(); predictor.train()"' > /app/scripts/start-model-trainer.sh \
    && echo '#!/bin/bash\necho "Starting dashboard..."\nstreamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0' > /app/scripts/start-dashboard.sh \
    && chmod +x /app/scripts/start-*.sh

# Create health check script
RUN echo '#!/bin/bash\nif [[ -f /app/data/health_status.txt ]]; then\n  exit 0\nelse\n  exit 1\nfi' > /app/scripts/healthcheck.sh \
    && chmod +x /app/scripts/healthcheck.sh

# Switch to non-root user
USER appuser

# Create health status file
RUN touch /app/data/health_status.txt

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD /app/scripts/healthcheck.sh
