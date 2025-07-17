FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_ml.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY training_data/ ./training_data/

# Create necessary directories
RUN mkdir -p input output models

# Set environment variables
ENV PYTHONPATH=/app/src

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "src/ml_pdf_extractor_v2.py", "--help"]
