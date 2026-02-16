# 🚨 Hazard Detection: Stairs & Other Hazards

## TL;DR - The Answer

**Q: Do you need to change YOLO version for stairs detection?**

**A: NO!** Your VLM (Qwen-VL) already can detect stairs and all hazards. You just need to:
1. ✅ Enhance VLM prompts to focus on hazards
2. ✅ Add VLM hazard checking to your fusion engine
3. ✅ Update safety rules to use VLM for hazards YOLO can't see

**Time needed**: 30 minutes  
**New models needed**: None  
**Training needed**: None  

---

## 🔍 The Problem

### Standard YOLO (YOLOv8/v11) Cannot Detect Stairs

**Why?** YOLO is trained on COCO dataset which has only 80 classes:
- ✅ person, car, bus, truck, bicycle, dog, cat, knife, etc.
- ❌ stairs, holes, edges, cliffs, wet floors, fire, guns, etc.

### Your Current Safety Rules Reference Objects YOLO Can't See

```python
# From perception_layer/rules.py
CRITICAL_OBJECTS = {
    "knife",    # ✅ YOLO CAN detect
    "gun",      # ❌ YOLO CANNOT detect
    "fire",     # ❌ YOLO CANNOT detect  
    "stairs",   # ❌ YOLO CANNOT detect ⚠️
    "hole",     # ❌ YOLO CANNOT detect ⚠️
    "car",      # ✅ YOLO CAN detect
    "edge",     # ❌ YOLO CANNOT detect ⚠️
}
```

**Result**: Safety rules won't trigger for stairs/holes/edges!

---

## ✅ The Solution: Use Your VLM!

### Your VLM (Qwen-VL) CAN Already Detect:
- ✅ Stairs (up or down)
- ✅ Holes, manholes, gaps
- ✅ Edges, cliffs, drop-offs
- ✅ Wet/slippery surfaces
- ✅ Fire, smoke
- ✅ ANY visual hazard

### How It Works:
```
Camera Frame
    ↓
YOLO (fast) → Detects: person, car, dog, etc.
    ↓
VLM (smart) → Sees EVERYTHING including stairs, holes, edges
    ↓
Fusion Engine → Combines both + checks for hazards
    ↓
User Alert: "Danger! Stairs detected ahead!"
```

---

## 🔧 Implementation (3 Simple Steps)

### Step 1: Enhance VLM Prompts for Hazard Detection

**File**: `reasoning_layer/vlm.py`  
**Lines**: ~99 and ~156

**Current prompt**:
```python
prompt = "Describe this scene briefly for a visually impaired person. Focus on obstacles, people, and navigation hazards. Keep it under 30 words."
```

**Enhanced prompt**:
```python
prompt = """Describe this scene for a visually impaired person walking forward.
CRITICAL - Check for navigation hazards:
- Stairs (up/down), steps, escalators
- Holes, manholes, gaps, pits
- Edges, cliffs, drop-offs, ledges
- Wet/slippery surfaces, water
- Obstacles blocking the path
- Fire, smoke, dangerous objects
Also mention: people, vehicles nearby. Max 45 words."""
```

### Step 2: Add VLM Hazard Detection Method

**File**: `fusion_layer/engine.py`  
**Add this new method**:

```python
def check_vlm_hazards(self, description: str) -> tuple[str, str] | None:
    """
    Analyzes VLM description for navigation hazards.
    Returns (alert_type, message) or None.
    """
    desc_lower = description.lower()
    
    # Critical hazards - immediate danger
    critical_patterns = [
        ("stair", "stairs"),
        ("step", "steps"),
        ("hole", "hole or gap"),
        ("manhole", "manhole"),
        ("edge", "edge or drop-off"),
        ("cliff", "cliff"),
        ("drop", "drop-off"),
        ("ledge", "ledge"),
        ("fire", "fire"),
        ("flame", "flames"),
    ]
    
    for keyword, display_name in critical_patterns:
        if keyword in desc_lower:
            return ("CRITICAL_ALERT", 
                   f"Danger! {display_name.capitalize()} detected ahead. Stop immediately.")
    
    # Warning hazards - proceed with caution
    warning_patterns = [
        ("wet", "wet surface"),
        ("slippery", "slippery surface"),
        ("puddle", "puddle"),
        ("water", "water on path"),
        ("obstacle", "obstacle"),
        ("blocked", "blocked path"),
        ("narrow", "narrow passage"),
    ]
    
    for keyword, display_name in warning_patterns:
        if keyword in desc_lower:
            return ("WARNING", 
                   f"Warning! {display_name.capitalize()} ahead. Proceed carefully.")
    
    return None
```

### Step 3: Integrate VLM Hazard Checking in Main Loop

**File**: `scripts/run_enhanced_camera.py`  
**Location**: After line ~526 where VLM result is received

**Add this code**:
```python
if result:
    new_desc, duration = result
    logger.info(f"VLM Description: {new_desc}")
    description = new_desc
    
    # 🆕 CHECK VLM DESCRIPTION FOR HAZARDS
    vlm_hazard = fusion.check_vlm_hazards(new_desc)
    if vlm_hazard:
        alert_type, message = vlm_hazard
        logger.warning(f"VLM Hazard Detected: {message}")
        fusion.handle_safety_alert(message, alert_type)
        description = f"ALERT: {message}"
    
    # Rest of existing code...
```

---

## 📊 Performance Comparison

| Detection Method | Speed | Can Detect Stairs? | Can Detect Holes? | Effort |
|-----------------|-------|-------------------|-------------------|--------|
| **YOLO only** | ⚡ 20-50ms | ❌ NO | ❌ NO | None |
| **VLM only** | 🐌 100-500ms | ✅ YES | ✅ YES | Low |
| **YOLO + VLM (recommended)** | Both | ✅ YES | ✅ YES | **30 min** ✅ |
| **Fine-tune YOLO** | ⚡ 20-50ms | ✅ YES | ⚠️ Maybe | 2-4 hours |
| **Depth model** | 🐌 200-800ms | ✅ YES | ✅ YES | 1-2 days |

---

## 🎯 Recommendation

**Use YOLO + VLM (Option 1)** because:

1. ✅ **Already works** - No new models needed
2. ✅ **Comprehensive** - Detects ALL visual hazards, not just stairs
3. ✅ **Quick** - 30 minutes to implement
4. ✅ **Reliable** - VLM sees everything YOLO misses
5. ✅ **Works with darkness detection** - Both systems complement each other

---

## 🚀 Alternative: Fine-tune YOLO (If You Want Faster Detection)

If you need faster stairs detection (20-50ms instead of 100-500ms):

### Option: Train YOLO on Stairs Dataset

```bash
# 1. Install Roboflow
pip install roboflow

# 2. Download stairs dataset (free datasets available)
# Visit: https://universe.roboflow.com/search?q=stairs

# 3. Fine-tune YOLO
from ultralytics import YOLO
model = YOLO('models/yolo/yolo11m.pt')
model.train(data='stairs-dataset/data.yaml', epochs=50, imgsz=640)

# 4. Save as: models/yolo/yolo11m_stairs.pt

# 5. Update config.json:
{
    "detector": {
        "active_model": "yolo11m_stairs",
        "models": {
            "yolo11m": "models/yolo/yolo11m.pt",
            "yolo11m_stairs": "models/yolo/yolo11m_stairs.pt"
        }
    }
}
```

**Time**: 2-4 hours  
**Benefit**: Faster stairs detection  
**Downside**: Only detects stairs, not other hazards  

---

## ✅ Summary

### What You Should Do:

1. **Implement VLM hazard detection** (30 minutes)
   - Enhance VLM prompts
   - Add hazard checking method
   - Integrate in main loop

2. **Test with real stairs**
   - Point camera at stairs
   - Should hear: "Danger! Stairs detected ahead!"

3. **(Optional) Fine-tune YOLO** if you need faster detection

### What You DON'T Need:

- ❌ Change YOLO version
- ❌ Download new models
- ❌ Train from scratch
- ❌ Complex depth estimation

---

## 🎬 Ready to Implement?

I can implement the VLM hazard detection for you right now. It will:
- ✅ Detect stairs, holes, edges, and all hazards
- ✅ Alert you immediately when detected
- ✅ Work alongside your existing YOLO detection
- ✅ Integrate with your darkness detection

**Shall I proceed with the implementation?**
