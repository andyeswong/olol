#!/bin/bash
# OLOL Quick Setup Script

echo "🚀 OLOL Quick Setup"
echo "=================="

# Check Python
echo "📋 Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi
python3 --version

# Check Ollama
echo "📋 Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi
ollama --version

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Install PyTorch with GPU support
echo "🔥 Installing PyTorch with CUDA support..."
# For RTX 5090 (Blackwell), use nightly builds with CUDA 12.8
if lspci | grep -i "rtx 5090\|rtx 5080\|rtx 5070" > /dev/null 2>&1; then
    echo "   🎯 RTX 50 series detected - installing nightly PyTorch with CUDA 12.8"
    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    echo "   🔧 Using stable PyTorch with CUDA 12.4"
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# Install OLOL package
echo "📦 Installing OLOL..."
python3 -m pip install -e .

# Check if models exist
echo "📋 Checking models..."
MODEL_COUNT=$(ollama list | wc -l)
if [ $MODEL_COUNT -le 1 ]; then
    echo "📥 No models found. Downloading a test model..."
    ollama pull llama3.2:3b
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "from src.olol.proto import ollama_pb2; print('✅ OLOL imports working')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start OLOL:"
echo "  Terminal 1: python3 -m olol server"
echo "  Terminal 2: python3 -m olol proxy --servers localhost:50051"
echo ""
echo "📖 For detailed setup, see DEPLOYMENT_GUIDE.md"