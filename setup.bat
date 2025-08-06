@echo off
REM OLOL Quick Setup Script for Windows

echo 🚀 OLOL Quick Setup
echo ==================

REM Check Python
echo 📋 Checking Python...
py --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)
py --version

REM Check Ollama
echo 📋 Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not found. Please download from https://ollama.ai/download/windows
    pause
    exit /b 1
)
ollama --version

REM Install Python dependencies
echo 📦 Installing Python dependencies...
py -m pip install -r requirements.txt

REM Install OLOL package
echo 📦 Installing OLOL...
py -m pip install -e .

REM Check if models exist
echo 📋 Checking models...
for /f %%i in ('ollama list ^| find /c /v ""') do set MODEL_COUNT=%%i
if %MODEL_COUNT% leq 1 (
    echo 📥 No models found. Downloading a test model...
    ollama pull llama3.2:3b
)

REM Test installation
echo 🧪 Testing installation...
py -c "from src.olol.proto import ollama_pb2; print('✅ OLOL imports working')"

echo.
echo ✅ Setup complete!
echo.
echo 🚀 To start OLOL:
echo   Terminal 1: py -m olol server
echo   Terminal 2: py -m olol proxy --servers localhost:50051
echo.
echo 📖 For detailed setup, see DEPLOYMENT_GUIDE.md
pause