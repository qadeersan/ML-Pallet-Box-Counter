# Pallet Box Counter - Production Dockerfile for GCP Cloud Run
# 
# Build: docker build -t pallet-box-counter -f docker/Dockerfile .
# Run Local: docker run -p 8000:8000 pallet-box-counter
# 
# For GCP Cloud Run deployment:
# gcloud builds submit --tag gcr.io/PROJECT_ID/pallet-box-counter
# gcloud run deploy pallet-box-counter --image gcr.io/PROJECT_ID/pallet-box-counter

# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Port for Cloud Run (Cloud Run sets PORT env var)
    PORT=8000 \
    # Disable GPU for Cloud Run (CPU only)
    CUDA_VISIBLE_DEVICES=""

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for production)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/
COPY data/combined_data.yaml ./data/

# Create non-root user for security (GCP Cloud Run best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (informational - Cloud Run uses PORT env var)
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the FastAPI application
# Cloud Run requires listening on 0.0.0.0 and PORT from environment
CMD exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
