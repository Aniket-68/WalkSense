#!/bin/bash
# WalkSense Quick Setup Script

set -e  # Exit on error

echo "========================================="
echo "WalkSense Enhanced Setup"
echo "========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi

# Create models directory
echo ""
echo "Creating directories..."
mkdir -p models/yolo
mkdir -p data
mkdir -p logs
mkdir -p docs
echo -e "${GREEN}✓ Directories created${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo -e "${YELLOW}Please activate it with: source venv/bin/activate${NC}"
else
    echo ""
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Install requirements
echo ""
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found, installing manually...${NC}"
    pip install torch torchvision opencv-python numpy ultralytics transformers pyttsx3 SpeechRecognition
fi

# Download YOLO model
echo ""
echo "Downloading YOLOv8n model..."
if [ ! -f "models/yolo/yolov8n.pt" ]; then
    python -c "
from ultralytics import YOLO
import shutil
print('Downloading YOLOv8n...')
model = YOLO('yolov8n.pt')
shutil.move('yolov8n.pt', 'models/yolo/yolov8n.pt')
print('Model downloaded to models/yolo/yolov8n.pt')
"
    echo -e "${GREEN}✓ YOLO model downloaded${NC}"
else
    echo -e "${YELLOW}YOLO model already exists${NC}"
fi

# Test camera
echo ""
echo "Testing camera..."
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if ret:
    print('Camera OK')
else:
    print('WARNING: Camera test failed')
" && echo -e "${GREEN}✓ Camera working${NC}" || echo -e "${RED}✗ Camera not working${NC}"

# Test YOLO
echo ""
echo "Testing YOLO detector..."
python -c "
from safety.yolo_detector import YoloDetector
detector = YoloDetector('models/yolo/yolov8n.pt')
print('YOLO detector initialized successfully')
" && echo -e "${GREEN}✓ YOLO working${NC}" || echo -e "${RED}✗ YOLO test failed${NC}"

# Test TTS
echo ""
echo "Testing TTS..."
python -c "
from audio.tts import TTSEngine
tts = TTSEngine()
print('TTS initialized successfully')
" && echo -e "${GREEN}✓ TTS working${NC}" || echo -e "${RED}✗ TTS test failed${NC}"

# Summary
echo ""
echo "========================================="
echo "Setup Summary"
echo "========================================="
echo -e "${GREEN}✓ Python environment ready${NC}"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo -e "${GREEN}✓ YOLO model downloaded${NC}"
echo ""
echo "Next steps:"
echo "1. Install and run LM Studio: https://lmstudio.ai"
echo "   - Download Qwen2-VL-2B-Instruct-GGUF"
echo "   - Start local server on port 1234"
echo ""
echo "2. Run the enhanced system:"
echo "   python scripts/run_enhanced_camera.py"
echo ""
echo "3. Or run basic demo (no LLM needed):"
echo "   python scripts/run_camera.py"
echo ""
echo "See SETUP_GUIDE.md for detailed instructions"
echo "========================================="
