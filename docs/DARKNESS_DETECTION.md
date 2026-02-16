# Darkness Detection Feature

## Overview
This feature prevents the WalkSense system from triggering VLM (Vision-Language Model) processing when 75% or more of the camera view is too dark. When darkness is detected, the system notifies the user and skips expensive VLM inference.

## Implementation Details

### 1. DarknessDetector Class
**Location:** `infrastructure/darkness_detector.py`

**Key Parameters:**
- `darkness_threshold`: Pixel brightness value below which is considered "dark" (default: 40 on 0-255 scale)
- `area_threshold`: Fraction of the image that must be dark to trigger (default: 0.75 = 75%)

**Methods:**
- `is_too_dark(frame)`: Returns `(bool, float)` indicating if frame is too dark and the percentage of dark pixels
- `get_brightness_level(frame)`: Returns human-readable brightness description

### 2. Integration in Main Loop
**Location:** `scripts/run_enhanced_camera.py`

**Changes Made:**
1. Import the `DarknessDetector` class
2. Initialize detector with 75% threshold: `DarknessDetector(darkness_threshold=40, area_threshold=0.75)`
3. Check darkness before VLM processing
4. Skip VLM and notify user when too dark
5. Clear pending queries if environment is too dark

**Workflow:**
```
Frame Captured
    ↓
Darkness Check (is 75%+ dark?)
    ↓
YES → Skip VLM, Notify User ("Your view is too dark...")
NO  → Proceed with normal VLM processing
```

### 3. User Notifications
When darkness is detected:
- **Visual**: Description panel shows "DARK ENVIRONMENT: [level] - Your view is too dark..."
- **Audio**: TTS speaks the darkness message
- **Frequency**: Alerts are rate-limited to once every 10 seconds to avoid spam
- **Query Handling**: If user has a pending query but it's too dark, the query is cleared and user is notified

## Technical Approach

### Darkness Detection Algorithm
1. Convert frame to grayscale for brightness analysis
2. Count pixels with brightness < threshold (40)
3. Calculate percentage of dark pixels
4. Compare against area threshold (75%)
5. Return result and percentage

### Brightness Levels
- **Very Dark**: Average brightness < 30
- **Dark**: Average brightness < 60
- **Dim**: Average brightness < 120
- **Normal**: Average brightness < 180
- **Bright**: Average brightness ≥ 180

## Testing

### Unit Tests
**Location:** `tests/test_darkness_detector.py`

Tests cover:
- Completely dark frame (100% dark) → Should trigger
- Completely bright frame (0% dark) → Should NOT trigger
- 75% dark frame (boundary) → Should trigger
- 50% dark frame → Should NOT trigger
- Dim frame (all pixels at threshold)

**Run tests:**
```bash
python -m tests.test_darkness_detector
```

### Visual Demonstration
**Location:** `tests/demo_darkness_detection.py`

Creates sample frames showing:
1. Normal bright scene (OK)
2. Dim scene (OK)
3. 50% dark (OK - below threshold)
4. 75% dark (BLOCKED - at threshold)
5. 90% dark (BLOCKED)
6. Completely dark (BLOCKED)

**Run demo:**
```bash
python -m tests.demo_darkness_detection
```

## Benefits

1. **Performance**: Saves GPU/CPU resources by skipping VLM when it can't see anything useful
2. **User Experience**: Provides clear feedback about why the system isn't responding
3. **Battery Life**: Reduces power consumption by avoiding unnecessary inference
4. **Accuracy**: Prevents hallucinations from VLM trying to describe a dark/black frame

## Configuration

The thresholds can be adjusted in the main script initialization:

```python
darkness_detector = DarknessDetector(
    darkness_threshold=40,    # Adjust pixel brightness threshold (0-255)
    area_threshold=0.75       # Adjust percentage threshold (0.0-1.0)
)
```

**Recommendations:**
- Lower `darkness_threshold` (e.g., 30) for stricter darkness detection
- Raise `area_threshold` (e.g., 0.85) to require more of the frame to be dark
- Lower `area_threshold` (e.g., 0.60) to be more sensitive to darkness

## Example Output

When darkness is detected:
```
[2026-01-31 23:28:15] WARNING - Darkness detected: 87.3% - Skipping VLM
[TTS] SPEAKING: Your view is too dark (87% dark). Please move to a brighter area.
```

When lighting returns to normal:
```
[2026-01-31 23:28:45] INFO - Normal lighting detected - Resuming VLM processing
```

## Future Enhancements

Potential improvements:
1. Adaptive thresholds based on time of day
2. Gradual transitions instead of hard cutoffs
3. Low-light VLM mode with adjusted prompts
4. Night vision mode using infrared if available
5. Historical brightness tracking to detect sudden changes
