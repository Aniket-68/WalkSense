# Hazard Detection Analysis & Solutions

## Current Situation

### ❌ Problem: Standard YOLO Cannot Detect Stairs and Many Hazards

**Your current model**: YOLOv11m trained on COCO dataset

**COCO Dataset Classes** (80 total):
- ✅ **CAN detect**: person, car, bus, truck, motorcycle, bicycle, dog, cat, knife, etc.
- ❌ **CANNOT detect**: stairs, holes, manholes, edges, cliffs, wet floors, obstacles, curbs, etc.

### What's in Your Safety Rules vs. What YOLO Can Actually Detect

```python
# From perception_layer/rules.py
CRITICAL_OBJECTS = {
    "knife",        # ✅ YOLO can detect
    "gun",          # ❌ YOLO CANNOT detect (not in COCO)
    "fire",         # ❌ YOLO CANNOT detect (not in COCO)
    "stairs",       # ❌ YOLO CANNOT detect (not in COCO) ⚠️
    "hole",         # ❌ YOLO CANNOT detect (not in COCO) ⚠️
    "car",          # ✅ YOLO can detect
    "bus",          # ✅ YOLO can detect
    "truck",        # ✅ YOLO can detect
    "edge",         # ❌ YOLO CANNOT detect (not in COCO) ⚠️
    "cliff"         # ❌ YOLO CANNOT detect (not in COCO) ⚠️
}
```

**Result**: Your safety rules reference objects that YOLO cannot detect!

---

## Solutions (Ranked by Difficulty)

### 🟢 Solution 1: Use VLM for Hazard Detection (EASIEST - Already Implemented!)

**Status**: ✅ **Already working in your system!**

**How it works**:
- YOLO detects common objects (person, car, etc.)
- VLM (Qwen-VL) sees the **entire scene** and can identify:
  - Stairs
  - Holes
  - Edges
  - Wet floors
  - Uneven surfaces
  - Any visual hazard

**Advantages**:
- ✅ No model changes needed
- ✅ Already implemented
- ✅ Can detect ANY visual hazard
- ✅ Provides context and descriptions

**Disadvantages**:
- ⚠️ Slower than YOLO (100-500ms vs 20-50ms)
- ⚠️ Requires good lighting (now handled by darkness detection!)
- ⚠️ May miss fast-moving hazards

**Recommendation**: **This is your best current solution**. Your VLM already handles stairs and hazards!

---

### 🟡 Solution 2: Fine-tune YOLO on Custom Hazard Dataset (MODERATE)

**What to do**:
1. Collect/download dataset with stairs, holes, edges, etc.
2. Fine-tune YOLOv11m on this custom dataset
3. Replace the model in your config

**Datasets available**:
- **Stair Detection**: [Staircase Dataset on Roboflow](https://universe.roboflow.com/search?q=stairs)
- **Obstacle Detection**: Various datasets on Roboflow Universe
- **Custom**: Record your own footage and label it

**Steps**:
```bash
# 1. Install Roboflow
pip install roboflow

# 2. Download a stairs dataset
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("stairs-detection")
dataset = project.version(1).download("yolov11")

# 3. Fine-tune YOLO
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
model.train(data='stairs-detection/data.yaml', epochs=50)

# 4. Update config.json
# "detector": {
#     "active_model": "yolo11m_stairs",
#     "models": {
#         "yolo11m_stairs": "models/yolo/yolo11m_stairs.pt"
#     }
# }
```

**Time required**: 2-4 hours (dataset + training)

**Advantages**:
- ✅ Fast detection (20-50ms)
- ✅ Real-time performance
- ✅ Specific to your needs

**Disadvantages**:
- ⚠️ Requires training data
- ⚠️ Need to retrain for new hazard types
- ⚠️ May not generalize well

---

### 🔴 Solution 3: Use Specialized Models (ADVANCED)

**Option A: Depth Estimation + Edge Detection**
- Use MiDaS or ZoeDepth for depth maps
- Detect stairs/edges from depth discontinuities
- Combine with YOLO for objects

**Option B: Segmentation Models**
- Use YOLOv11-seg (segmentation variant)
- Segment floor, stairs, obstacles
- More accurate spatial understanding

**Option C: Multi-Model Ensemble**
- YOLO for objects
- Depth model for stairs/edges
- VLM for context and verification

**Time required**: 1-2 days

---

## 🎯 Recommended Approach

### Hybrid Strategy (Best of All Worlds)

```
┌─────────────────────────────────────────────────────┐
│                  Camera Frame                        │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
   ┌─────────┐         ┌──────────┐
   │  YOLO   │         │   VLM    │
   │ (Fast)  │         │ (Smart)  │
   └────┬────┘         └────┬─────┘
        │                   │
        │ Common objects    │ Hazards (stairs, holes, etc.)
        │ (person, car)     │ + Scene context
        │                   │
        └──────────┬────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Fusion Engine  │
          │ + Safety Rules │
          └────────────────┘
```

**Implementation**:
1. ✅ **Keep YOLO** for fast common object detection
2. ✅ **Keep VLM** for hazard detection (stairs, holes, edges)
3. ✅ **Update Safety Rules** to rely on VLM for hazards
4. ✅ **Add VLM-specific prompts** for hazard detection

---

## 🔧 Immediate Action Items

### 1. Update Safety Rules to Separate YOLO vs VLM Detection

<parameter name="Complexity">7
