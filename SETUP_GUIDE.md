# WalkSense Setup Guide

## System Requirements

### Hardware
- **CPU**: Quad-core or better (Intel i5/AMD Ryzen 5+)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended for VLM)
- **Camera**: USB webcam or laptop camera
- **Microphone**: For speech input (optional)
- **Speakers**: For TTS output

### Operating System
- **Windows**: 10/11 (tested)
- **Linux**: Ubuntu 20.04+ (recommended)
- **macOS**: 11+ (limited testing)

---

## Step-by-Step Setup

### 1. Clone Repository

```bash
cd ~
git clone <your-repo-url>
cd WalkSense
```

### 2. Create Python Environment

#### Using venv (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

#### Or using conda
```bash
conda create -n walksense python=3.10
conda activate walksense
```

### 3. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# If you get errors, install in stages:

# 1. Core ML/DL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Vision processing
pip install opencv-python==4.8.0 Pillow==10.0.0 numpy==1.24.0

# 3. YOLO
pip install ultralytics==8.2.0

# 4. VLM (Qwen)
pip install transformers==4.41.0 accelerate==0.28.0 peft==0.11.0

# 5. Audio (TTS/STT)
pip install pyttsx3==2.90 SpeechRecognition==3.10.0

# 6. Utilities
pip install pydantic==2.6.0 pyyaml==6.0.1 loguru==0.7.2 tqdm rich

# 7. Additional dependencies for Qwen VLM
pip install qwen-vl-utils
```

### 4. Download YOLO Model

```bash
# Create models directory
mkdir -p models/yolo

# Download YOLOv8n using Python
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLOv8n downloaded successfully')
"

# Move model to correct location
# On Windows:
move yolov8n.pt models\yolo\yolov8n.pt
# On Linux/Mac:
mv yolov8n.pt models/yolo/yolov8n.pt
```

Verify YOLO installation:
```bash
python -c "
from safety.yolo_detector import YoloDetector
detector = YoloDetector('models/yolo/yolov8n.pt')
print('YOLO detector initialized successfully')
"
```

### 5. Setup LM Studio (for VLM and LLM)

#### Download LM Studio
1. Go to [lmstudio.ai](https://lmstudio.ai/)
2. Download for your OS
3. Install and launch

#### Download Vision Model

**Option A: Qwen2-VL-2B (Recommended - fits in 4GB VRAM)**
1. In LM Studio, go to **Search** tab
2. Search for: `Qwen2-VL-2B-Instruct-GGUF`
3. Download: `qwen2-vl-2b-instruct-q4_k_m.gguf` (smaller) or `q5_k_m` (better quality)
4. Click **Load Model** to load into memory

**Option B: Qwen2-VL-7B (Better quality - needs 8GB+ VRAM)**
1. Search for: `Qwen2-VL-7B-Instruct-GGUF`
2. Download quantized version (Q4 or Q5)
3. Load into LM Studio

#### Configure LM Studio API Server

1. Click **Local Server** tab (left sidebar)
2. Settings:
   - **Port**: `1234` (default)
   - **CORS**: Enable if running from browser
   - **GPU Offload**: Set to max layers your GPU can handle
3. Click **Start Server**
4. Verify API is running: Open browser to `http://localhost:1234/v1/models`

**Expected response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2-vl-2b-instruct",
      "object": "model",
      ...
    }
  ]
}
```

### 6. Test Vision Model

```bash
python -c "
from reasoning.qwen_vlm import QwenVLM
import numpy as np

# Test LM Studio connection
qwen = QwenVLM(backend='lm_studio', lm_studio_url='http://localhost:1234/v1')

# Create dummy frame
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Test inference
desc = qwen.describe_scene(dummy_frame, context='test')
print(f'VLM Response: {desc}')
"
```

### 7. Load Text LLM for Query Answering

You need **TWO models running**:
1. **Vision model** (Qwen2-VL) - for scene description
2. **Text model** (any LLM) - for query answering

#### Load Second Model (for LLM Reasoner)

**Option 1: Use same Qwen2-VL model (simpler)**
- Already loaded, no extra setup needed

**Option 2: Load separate text model (recommended)**
1. In LM Studio, click **+** to load another model
2. Recommended text models:
   - **Llama-3-8B-Instruct** (good balance)
   - **Mistral-7B-Instruct** (fast)
   - **Phi-3-mini** (very fast, smaller)
3. Load in LM Studio alongside Qwen2-VL

> **Note**: LM Studio can multiplex requests to different loaded models via the same API endpoint.

### 8. Configure Camera

#### Test Camera
```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f'Camera working! Resolution: {frame.shape[1]}x{frame.shape[0]}')
else:
    print('ERROR: Cannot access camera')
cap.release()
"
```

#### Fix Camera Issues

**Windows:**
- Ensure camera permissions enabled in Settings → Privacy → Camera
- Try different camera IDs: `cv2.VideoCapture(0)`, `cv2.VideoCapture(1)`, etc.

**Linux:**
```bash
# Check available cameras
ls /dev/video*

# Give permissions
sudo usermod -a -G video $USER
# Log out and back in
```

**macOS:**
- Grant Terminal/IDE camera permissions in System Preferences → Security & Privacy

### 9. Test TTS (Text-to-Speech)

```bash
python -c "
from audio.tts import TTSEngine
tts = TTSEngine()
tts.speak('WalkSense is ready')
"
```

**Fix TTS Issues:**

**Windows:**
- pyttsx3 should work out of the box

**Linux:**
```bash
# Install espeak
sudo apt-get install espeak

# Test
espeak "Hello world"
```

**macOS:**
- Uses built-in say command (should work)

### 10. Test STT (Speech-to-Text) - Optional

```bash
# Install PyAudio for microphone
# Windows:
pip install pyaudio

# Linux:
sudo apt-get install portaudio19-dev
pip install pyaudio

# macOS:
brew install portaudio
pip install pyaudio

# Test STT
python -c "
from interaction.stt_listener import STTListener
listener = STTListener()
print('Say something...')
text = listener.listen_once(timeout=5)
print(f'You said: {text}')
"
```

---

## Running the System

### Option 1: Basic Demo (Original)

```bash
python scripts/run_camera.py
```

**Controls:**
- `S` - Start system
- `M` - Mute/unmute audio
- `Q` - Quit

### Option 2: Enhanced Demo (with Spatial Tracking + LLM)

```bash
python scripts/run_enhanced_camera.py
```

**Controls:**
- `S` - Start system
- `L` - Ask question (push-to-talk for STT)
- `M` - Mute/unmute audio
- `Q` - Quit

**Expected workflow:**
1. Press `S` to start
2. System announces detected objects
3. VLM provides scene descriptions every ~5 seconds
4. Press `L` to ask questions like:
   - "What's in front of me?"
   - "Is it safe to cross?"
   - "What's on my left?"
5. LLM answers using spatial context + scene understanding

---

## Configuration

### Edit Configuration in `run_enhanced_camera.py`

```python
# VLM Backend (lines 113-120)
QWEN_BACKEND = "lm_studio"  # or "huggingface"
LM_STUDIO_URL = "http://localhost:1234/v1"  # Change if LM Studio on different machine
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# LLM Backend (lines 122-123)
LLM_BACKEND = "lm_studio"
LLM_URL = "http://localhost:1234/v1"  # Same as VLM if using same LM Studio instance
```

### Tune Performance

```python
# Frame sampling rate (line 143)
sampler = FrameSampler(every_n_frames=150)  # Lower = more frequent VLM calls

# Scene change sensitivity (line 212)
scene_detector = SceneChangeDetector(threshold=0.15)  # Lower = more sensitive
```

### Spatial Tracking Sensitivity

Edit in `inference/fusion_engine.py`:
```python
self.spatial_context = SpatialContextManager(
    movement_threshold=30.0,  # Lower = announce smaller movements
    time_threshold=10.0       # Lower = announce more frequently
)
```

---

## Verification Checklist

Run through this checklist to ensure everything works:

### ✅ Core Components
```bash
# 1. YOLO
python -c "from safety.yolo_detector import YoloDetector; d=YoloDetector(); print('✓ YOLO OK')"

# 2. Camera
python -c "from safety.frame_capture import Camera; c=Camera(); print('✓ Camera OK')"

# 3. TTS
python -c "from audio.tts import TTSEngine; t=TTSEngine(); t.speak('Test'); print('✓ TTS OK')"

# 4. Safety Rules
python -c "from safety.safety_rules import SafetyRules; s=SafetyRules(); print('✓ Safety OK')"
```

### ✅ Enhanced Components
```bash
# 5. Spatial Context Manager
python -c "from inference.spatial_context_manager import SpatialContextManager; s=SpatialContextManager(); print('✓ Spatial Context OK')"

# 6. LLM Reasoner
python -c "from inference.llm_reasoner import LLMReasoner; l=LLMReasoner(backend='lm_studio'); print('✓ LLM OK')"

# 7. Fusion Engine
python -c "from inference.fusion_engine import FusionEngine; from audio.tts import TTSEngine; f=FusionEngine(TTSEngine()); print('✓ Fusion Engine OK')"
```

### ✅ LM Studio Connection
```bash
# Test API connectivity
curl http://localhost:1234/v1/models

# Should return JSON with loaded models
```

---

## Troubleshooting

### Issue: "LM Studio error: Connection refused"

**Solution:**
1. Ensure LM Studio is running
2. Click "Start Server" in Local Server tab
3. Verify port is 1234
4. Check firewall isn't blocking localhost:1234

### Issue: "YOLO model not found"

**Solution:**
```bash
# Download to correct location
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mkdir -p models/yolo
mv yolov8n.pt models/yolo/
```

### Issue: "Camera not working"

**Solution:**
```python
# Try different camera IDs
# Edit in safety/frame_capture.py line 8:
self.cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Issue: "STT not working"

**Solution:**
1. Check microphone permissions
2. Install PyAudio (see step 10)
3. Test microphone in OS settings first

### Issue: "VLM too slow"

**Solutions:**
- Use smaller model (Qwen2-VL-2B instead of 7B)
- Increase `every_n_frames` to 200-300
- Reduce quality (Q4 instead of Q5 quantization)
- Enable GPU offload in LM Studio

### Issue: "Out of memory"

**Solutions:**
- Use CPU mode (slower): In LM Studio, reduce GPU layers to 0
- Close other applications
- Use smaller models
- Reduce resolution: Edit `Camera(width=640, height=480)`

### Issue: "Import errors"

**Solution:**
```bash
# Ensure working directory is WalkSense root
cd /path/to/WalkSense

# Try running from python -m
python -m scripts.run_enhanced_camera

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
```

### Issue: "qwen-vl-utils not found"

**Solution:**
```bash
pip install qwen-vl-utils

# If still fails, install from source:
pip install git+https://github.com/QwenLM/Qwen-VL.git
```

---

## Alternative: Running Without LLM (Simpler Setup)

If LM Studio setup is too complex, you can run basic mode:

### Use Original Demo
```bash
python scripts/run_camera.py
```

This runs:
- ✅ YOLO detection
- ✅ Safety rules
- ✅ VLM scene descriptions
- ❌ No LLM query answering
- ❌ No spatial tracking

---

## Alternative LLM Backends

### Use Ollama Instead of LM Studio

#### Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from ollama.com
```

#### Pull Models
```bash
# Vision model
ollama pull llava:7b

# Text model for queries
ollama pull llama3:8b
```

#### Configure
Edit `run_enhanced_camera.py`:
```python
LLM_BACKEND = "ollama"
LLM_URL = "http://localhost:11434"
```

---

## Minimal Working Example

For quick testing without full setup:

```python
# test_basic.py
import cv2
from safety.yolo_detector import YoloDetector

detector = YoloDetector('models/yolo/yolov8n.pt')
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    detections = detector.detect(frame)
    
    # Print detections
    for d in detections:
        print(f"{d['label']}: {d['confidence']:.2f}")
    
    # Display
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run:
```bash
python test_basic.py
```

---

## Next Steps

Once everything is working:

1. ✅ Run `python scripts/run_enhanced_camera.py`
2. ✅ Test asking questions with 'L' key
3. ✅ Tune thresholds for your use case
4. ✅ Read `docs/API_EXAMPLES.md` for customization
5. ✅ Check `ENHANCED_SYSTEM.md` for features

---

## Getting Help

If you encounter issues:

1. Check console output for error messages
2. Verify all checklist items above
3. Test components individually
4. Check GitHub issues (if public repo)
5. Review logs in console output

**Common log patterns:**

Good:
```
[FusionEngine] Initialized with spatial tracking + LLM reasoning
[QWEN] Connected to LM Studio successfully
[WalkSense] Enhanced system running
```

Bad:
```
[QWEN WARNING] Could not connect to LM Studio
[TTS ERROR] pyttsx3 initialization failed
[Camera] ERROR: Could not open camera 0
```
