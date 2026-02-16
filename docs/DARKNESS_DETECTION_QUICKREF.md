# Darkness Detection - Quick Reference

## What It Does
When 75% or more of your camera view is dark, WalkSense will:
- ✅ Skip VLM processing (saves resources)
- 🔊 Notify you: "Your view is too dark (XX% dark). Please move to a brighter area."
- 📝 Show status in the UI: "DARK ENVIRONMENT: [brightness level]"
- ⏸️ Clear any pending queries (they can't be answered in darkness)

## How It Works
1. Every frame is analyzed for brightness
2. Pixels with brightness < 40 (out of 255) are considered "dark"
3. If ≥75% of pixels are dark → VLM is blocked
4. Alerts are shown every 10 seconds (not spammed)

## Testing Your Implementation

### Run Unit Tests
```bash
python -m tests.test_darkness_detector
```
Expected output: All 5 tests should pass ✓

### Run Visual Demo
```bash
python -m tests.demo_darkness_detection
```
This creates 6 sample frames showing different darkness levels.

### Test in Real-Time
1. Start WalkSense: `python -m scripts.run_enhanced_camera`
2. Press 'S' to start the system
3. Cover your camera lens → Should see darkness alert
4. Uncover lens → System resumes normal operation

## Adjusting Sensitivity

In `scripts/run_enhanced_camera.py`, line ~381:
```python
darkness_detector = DarknessDetector(
    darkness_threshold=40,    # Lower = stricter (more sensitive)
    area_threshold=0.75       # Higher = requires more darkness to trigger
)
```

### Examples:
- **More sensitive**: `darkness_threshold=30, area_threshold=0.60`
- **Less sensitive**: `darkness_threshold=50, area_threshold=0.85`

## Troubleshooting

**Problem**: False positives (triggers when it shouldn't)
- **Solution**: Increase `darkness_threshold` or `area_threshold`

**Problem**: Not triggering when it should
- **Solution**: Decrease `darkness_threshold` or `area_threshold`

**Problem**: Too many alerts
- **Solution**: Alerts are already rate-limited to 10 seconds. Check line ~570 to adjust.

## Files Modified/Created

### New Files:
- `infrastructure/darkness_detector.py` - Core detection logic
- `infrastructure/__init__.py` - Package initialization
- `tests/test_darkness_detector.py` - Unit tests
- `tests/demo_darkness_detection.py` - Visual demonstration
- `docs/DARKNESS_DETECTION.md` - Full documentation

### Modified Files:
- `scripts/run_enhanced_camera.py` - Integrated darkness detection

## Performance Impact
- **Minimal**: Grayscale conversion and pixel counting is very fast (~1-2ms)
- **Benefit**: Saves 100-500ms per VLM call when skipped
- **Net Result**: Improved performance in dark environments
