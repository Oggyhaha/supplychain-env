# Dockerfile
# ═══════════════════════════════════════════════
# SUPPLYCHAIN-ENV CONTAINER
#
# Build:  docker build -t supplychain-env .
# Run:    docker run -p 7860:7860 supplychain-env
# ═══════════════════════════════════════════════

# Base image: Python 3.11 on slim Linux
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
# (Docker caches this layer if requirements unchanged)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Copy entire project into container
COPY . .

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health')" \
    || exit 1

# Start the server
CMD ["python", "server/app.py"]