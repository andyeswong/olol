# OLOL Deployment Guide

## Prerequisites

### 1. Install Python 3.8+
```bash
# Check Python version
python --version
# or
python3 --version
```

### 2. Install Ollama
```bash
# On Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows
# Download from: https://ollama.ai/download/windows
```

### 3. Verify Ollama Installation
```bash
ollama --version
ollama list  # Should show available models
```

## Setup Steps

### 1. Clone and Setup OLOL
```bash
git clone https://github.com/andyeswong/olol.git
cd olol

# Install dependencies
pip install -r requirements.txt
# or if using poetry
poetry install
```

### 2. Install OLOL Package
```bash
# Install in development mode
pip install -e .

# Verify installation
python -m olol --version
```

### 3. Download Models (Required)
```bash
# Download at least one model for testing
ollama pull codestral:22b
# or a smaller model for testing
ollama pull llama3.2:3b

# Verify models are available
ollama list
```

### 4. Generate Protocol Buffers (if needed)
```bash
# This usually happens automatically, but if needed:
python -c "from src.olol import *"
```

## Running OLOL

### Option 1: Basic Setup (Single Machine)
```bash
# Terminal 1: Start gRPC server
python -m olol server --host localhost --port 50051

# Terminal 2: Start HTTP proxy
python -m olol proxy --host localhost --port 8000 --servers "localhost:50051" --no-discovery
```

### Option 2: Production Setup (Multiple Machines)
```bash
# On Server Machine 1:
python -m olol server --host 0.0.0.0 --port 50051

# On Server Machine 2:
python -m olol server --host 0.0.0.0 --port 50051

# On Proxy Machine:
python -m olol proxy --host 0.0.0.0 --port 8000 --servers "server1:50051,server2:50051"
```

### Option 3: With Discovery (Auto-detection)
```bash
# On each server:
python -m olol server --host 0.0.0.0 --port 50051

# On proxy (will auto-discover servers):
python -m olol proxy --host 0.0.0.0 --port 8000
```

## Testing the Installation

### 1. Test gRPC Server
```bash
# Quick health check
python -c "
import grpc
from src.olol.proto import ollama_pb2_grpc, ollama_pb2
channel = grpc.insecure_channel('localhost:50051')
stub = ollama_pb2_grpc.OllamaServiceStub(channel)
health = stub.HealthCheck(ollama_pb2.HealthCheckRequest())
print(f'Health: {health.healthy}')
"
```

### 2. Test HTTP Proxy
```bash
# Test status
curl http://localhost:8000/api/status

# Test models
curl http://localhost:8000/api/models

# Test generation
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "codestral:22b", "prompt": "Hello", "stream": false}'
```

## Docker Setup (Alternative)

### 1. Create Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy OLOL code
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install -e .

# Expose ports
EXPOSE 50051 8000

# Start script
CMD ["python", "-m", "olol", "proxy", "--host", "0.0.0.0"]
```

### 2. Build and Run
```bash
docker build -t olol .
docker run -p 8000:8000 -p 50051:50051 olol
```

## Configuration Options

### Environment Variables
```bash
export OLOL_SERVER_HOST=0.0.0.0
export OLOL_SERVER_PORT=50051
export OLOL_PROXY_HOST=0.0.0.0
export OLOL_PROXY_PORT=8000
export OLOL_OLLAMA_HOST=http://localhost:11434
```

### Configuration File (olol.yaml)
```yaml
server:
  host: 0.0.0.0
  port: 50051
  ollama_host: http://localhost:11434

proxy:
  host: 0.0.0.0
  port: 8000
  servers:
    - localhost:50051
  enable_ui: true
  enable_discovery: true
```

## Troubleshooting

### Common Issues:

1. **Port already in use**
```bash
# Find process using port
netstat -tulpn | grep :50051
# Kill if needed
kill -9 <PID>
```

2. **No models available**
```bash
# Download a model
ollama pull llama3.2:3b
# Verify
ollama list
```

3. **gRPC connection refused**
```bash
# Check if server is running
netstat -tulpn | grep :50051
# Check firewall settings
```

4. **Import errors**
```bash
# Reinstall dependencies
pip install -e . --force-reinstall
```

## Performance Optimization

### 1. System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB+ RAM, 8+ CPU cores
- **GPU**: Optional but recommended for large models

### 2. Model Selection
```bash
# Small models (testing)
ollama pull llama3.2:1b    # ~1GB
ollama pull llama3.2:3b    # ~2GB

# Medium models (production)
ollama pull codestral:22b  # ~13GB
ollama pull llama3.1:8b    # ~4.7GB

# Large models (high-performance)
ollama pull llama3.1:70b   # ~40GB
```

### 3. Network Configuration
```bash
# For multiple machines, ensure firewall allows:
# - Port 50051 (gRPC server)
# - Port 8000 (HTTP proxy)
# - Port 7946 (discovery, if enabled)
```

## Production Checklist

- [ ] Python 3.8+ installed
- [ ] Ollama installed and working
- [ ] At least one model downloaded
- [ ] OLOL dependencies installed
- [ ] Firewall ports configured
- [ ] Health checks passing
- [ ] Models populated in proxy
- [ ] Generate functionality tested
- [ ] Monitoring/logging configured

## Support

For issues:
1. Check the logs: `python -m olol proxy --verbose`
2. Verify Ollama: `ollama list`
3. Test components individually
4. Check the GitHub repository for updates