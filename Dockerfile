FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY MANIFEST.in .

# Install OLOL package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 olol
USER olol

# Expose ports
EXPOSE 50051 8000

# Default command
CMD ["python", "-m", "olol", "server", "--host", "0.0.0.0"]