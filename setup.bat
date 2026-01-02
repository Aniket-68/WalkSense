@echo off
REM WalkSense Quick Setup Script for Windows

echo =========================================
echo WalkSense Enhanced Setup
echo =========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python from python.org
    pause
    exit /b 1
)

echo Python found!
python --version
echo.

REM Create directories
echo Creating directories...
if not exist "models\yolo" mkdir models\yolo
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "docs" mkdir docs
echo [OK] Directories created
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
    echo.
) else (
    echo Virtual environment already exists
    echo.
)

REM Activate venv
call venv\Scripts\activate
echo [OK] Virtual environment activated
echo.

REM Install requirements
echo Installing Python dependencies...
echo This may take several minutes...
if exist "requirements.txt" (
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [OK] Dependencies installed
) else (
    echo Warning: requirements.txt not found
    echo Installing core dependencies...
    pip install torch torchvision opencv-python numpy ultralytics transformers pyttsx3 SpeechRecognition
)
echo.

REM Download YOLO model
echo Downloading YOLOv8n model...
if not exist "models\yolo\yolov8n.pt" (
    python -c "from ultralytics import YOLO; import shutil; model = YOLO('yolov8n.pt'); shutil.move('yolov8n.pt', 'models/yolo/yolov8n.pt'); print('Model downloaded')"
    echo [OK] YOLO model downloaded
) else (
    echo [OK] YOLO model already exists
)
echo.

REM Test camera
echo Testing camera...
python -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); cap.release(); print('[OK] Camera working' if ret else '[ERROR] Camera failed')"
echo.

REM Test YOLO
echo Testing YOLO detector...
python -c "from safety.yolo_detector import YoloDetector; YoloDetector('models/yolo/yolov8n.pt'); print('[OK] YOLO working')"
echo.

REM Test TTS
echo Testing TTS...
python -c "from audio.tts import TTSEngine; TTSEngine(); print('[OK] TTS working')"
echo.

echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Install LM Studio from https://lmstudio.ai
echo    - Download Qwen2-VL-2B-Instruct-GGUF model
echo    - Start local server on port 1234
echo.
echo 2. Run the enhanced system:
echo    python scripts\run_enhanced_camera.py
echo.
echo 3. Or run basic demo (no LLM):
echo    python scripts\run_camera.py
echo.
echo See SETUP_GUIDE.md for detailed instructions
echo =========================================
echo.
pause
