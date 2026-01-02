# WalkSense Quick Start

## 🚀 Fast Setup (5 minutes)

### 1. Automated Setup

**Linux/Mac:**
```bash
cd WalkSense
./setup.sh
```

**Windows:**
```cmd
cd WalkSense
setup.bat
```

This will:
- ✅ Create virtual environment
- ✅ Install Python dependencies
- ✅ Download YOLO model
- ✅ Test camera and components

### 2. Install LM Studio

1. Download from **[lmstudio.ai](https://lmstudio.ai)**
2. Install and launch
3. Search for `Qwen2-VL-2B-Instruct-GGUF`
4. Download the model (Q4 or Q5 version)
5. Click **"Local Server"** tab → **"Start Server"**

### 3. Run WalkSense

**Enhanced version (with LLM):**
```bash
python scripts/run_enhanced_camera.py
```

**Basic version (no LLM needed):**
```bash
python scripts/run_camera.py
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `S` | **Start** system |
| `L` | **Ask question** (push-to-talk) |
| `M` | **Mute/Unmute** audio |
| `Q` | **Quit** |

---

## 🎯 Quick Test

After running, you should see:

1. **Camera window opens** showing live feed
2. **YOLO detections** with colored bounding boxes
3. **Spatial tracking** info at top (`Tracking: car center, person left`)
4. **Audio announcements** for detected objects

**Try:**
- Wave an object in front of camera → should announce "object detected"
- Press `L` and ask "What do you see?" → LLM answers with context

---

## 🔧 Common Issues

### "LM Studio connection failed"
→ Make sure LM Studio is running and server started on port 1234

### "Camera not working"
→ Check camera permissions, try different camera ID in code

### "YOLO model not found"
→ Run setup script again or manually download:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/yolo/
```

### "Import errors"
→ Make sure you're in WalkSense directory:
```bash
cd /path/to/WalkSense
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

---

## 📂 Project Structure

```
WalkSense/
├── scripts/
│   ├── run_camera.py              # Basic demo
│   └── run_enhanced_camera.py     # Enhanced with LLM
├── inference/
│   ├── spatial_context_manager.py # Object tracking (NEW)
│   ├── llm_reasoner.py           # LLM integration (NEW)
│   └── fusion_engine.py          # Enhanced pipeline
├── safety/
│   ├── yolo_detector.py          # YOLO detection
│   └── safety_rules.py           # Hazard classification
└── models/
    └── yolo/
        └── yolov8n.pt            # YOLO weights
```

---

## 🎓 Learn More

- **`SETUP_GUIDE.md`** - Detailed setup instructions
- **`ENHANCED_SYSTEM.md`** - Feature documentation
- **`docs/API_EXAMPLES.md`** - Code examples
- **`docs/PIPELINE_FLOW.md`** - Architecture diagrams

---

## 💡 Example Questions to Ask

Once running, press `L` and try:

- ❓ "What's in front of me?"
- ❓ "Is it safe to move forward?"
- ❓ "What's on my left?"
- ❓ "Describe what you see"
- ❓ "Are there any obstacles?"

The LLM will answer using:
- Current object positions (from spatial tracking)
- Scene understanding (from VLM)
- Recent movement events

---

## 🏗️ Architecture Overview

```
Camera → YOLO → Spatial Tracker → Context Manager
                     ↓
                VLM Scene Description
                     ↓
              Context Memory
                     ↓
User Query → LLM Reasoner → Answer → TTS
```

**Key Features:**
- 🎯 **Object Tracking**: Maintains identity across frames
- 📍 **Spatial Awareness**: Knows position, direction, distance
- 🧠 **LLM Reasoning**: Answers questions intelligently
- 🚨 **Safety First**: Critical alerts always override

---

## 🔄 Workflow Example

1. **Frame 1**: Camera detects car
   - 🎯 Tracker assigns ID: `track_42`
   - 📢 Announces: "car detected on center"

2. **Frame 50**: Same car moved
   - 🎯 Tracker: "car moved 45px"
   - 📢 Announces: "car moving center"

3. **User asks**: "Is it safe?"
   - 🧠 LLM gets context: "car center at close distance"
   - 🧠 LLM gets VLM: "Parking lot with vehicles"
   - 📢 Answers: "Caution, car close ahead"

---

## 📊 Performance

- **YOLO**: 30-50ms per frame
- **Spatial Tracking**: ~5ms (negligible)
- **VLM**: 2-5s (async, non-blocking)
- **LLM Query**: 1-3s

Total real-time loop: **~30-50ms** ✅

---

## 🎛️ Tuning

Edit in `run_enhanced_camera.py`:

```python
# Process VLM every N frames (default: 150)
sampler = FrameSampler(every_n_frames=150)

# Scene change threshold (default: 15%)
scene_detector = SceneChangeDetector(threshold=0.15)

# Spatial tracking sensitivity
SpatialContextManager(
    movement_threshold=30.0,  # pixels
    time_threshold=10.0       # seconds
)
```

**Lower values** = More sensitive, more announcements
**Higher values** = Less sensitive, fewer announcements

---

## 🆘 Need Help?

1. Check `SETUP_GUIDE.md` for detailed troubleshooting
2. Verify LM Studio is running
3. Test components individually (see setup scripts)
4. Check console for error messages
