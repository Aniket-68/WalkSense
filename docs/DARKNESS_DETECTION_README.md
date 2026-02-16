# 🌙 Darkness Detection Feature

## Quick Start

Your WalkSense system now automatically detects when the camera view is too dark and skips VLM processing to save resources.

### What Happens When It's Dark?
- 🚫 VLM processing is **skipped** (saves 100-500ms per frame)
- 🔊 You hear: **"Your view is too dark (XX% dark). Please move to a brighter area."**
- 📺 UI shows: **"DARK ENVIRONMENT: [brightness level]"**
- 🧹 Pending queries are **cleared** (can't be answered in darkness)

### Threshold
- Triggers when **≥75%** of the camera view is dark
- "Dark" = pixel brightness < 40 (on 0-255 scale)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [**SUMMARY**](DARKNESS_DETECTION_SUMMARY.md) | Complete implementation overview |
| [**FULL DOCS**](DARKNESS_DETECTION.md) | Technical details and configuration |
| [**QUICK REF**](DARKNESS_DETECTION_QUICKREF.md) | Testing and troubleshooting guide |

---

## 🧪 Testing

### Run Unit Tests
```bash
python -m tests.test_darkness_detector
```
Expected: All 5 tests pass ✓

### Run Visual Demo
```bash
python -m tests.demo_darkness_detection
```
Creates 6 sample frames showing different darkness levels.

### Test Live
1. Start: `python -m scripts.run_enhanced_camera`
2. Press `S` to start
3. Cover camera lens → See darkness alert
4. Uncover lens → Normal operation resumes

---

## ⚙️ Adjust Sensitivity

Edit `scripts/run_enhanced_camera.py` (line ~381):

```python
# Default (75% threshold)
darkness_detector = DarknessDetector(
    darkness_threshold=40,
    area_threshold=0.75
)

# More sensitive (60% threshold)
darkness_detector = DarknessDetector(
    darkness_threshold=50,
    area_threshold=0.60
)

# Less sensitive (85% threshold)
darkness_detector = DarknessDetector(
    darkness_threshold=30,
    area_threshold=0.85
)
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Darkness check overhead | ~1-2ms |
| VLM processing time saved | ~100-500ms |
| Alert rate limit | 10 seconds |
| Net performance gain | **Significant** in dark environments |

---

## 🎯 Benefits

✅ **Performance**: Skips expensive VLM when it can't see  
✅ **User Experience**: Clear feedback about darkness  
✅ **Accuracy**: Prevents VLM hallucinations on black frames  
✅ **Efficiency**: Saves GPU/CPU/battery resources  

---

## 📁 Files

### Created
- `infrastructure/darkness_detector.py` - Core logic
- `tests/test_darkness_detector.py` - Unit tests
- `tests/demo_darkness_detection.py` - Visual demo
- `docs/DARKNESS_DETECTION*.md` - Documentation

### Modified
- `scripts/run_enhanced_camera.py` - Integration

---

## ✅ Status

**COMPLETE AND TESTED** ✓

All tests passing, documentation complete, ready for use!
