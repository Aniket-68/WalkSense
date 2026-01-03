# WalkSense - Navigation Mode Configuration

## ✅ Changes Made for Human-Friendly Navigation

### 1. **Reduced Announcement Frequency**
- Movement threshold: 30px → **80px** (only announce significant movements)
- Time cooldown: 10s → **15s** (like GPS navigation systems)
- Safety cooldown: 10s → **20s** (less repetitive warnings)

### 2. **Smarter Object Filtering**
- **Critical alerts** (immediate danger): Only fire, stairs
- **Warning alerts** (navigation hazards): Cars, trucks, buses, people
- **Removed**: knives, guns, motorcycles, bicycles, dogs, books, phones, etc.

### 3. **Less Frequent Scene Descriptions (VLM)**
- VLM sampling: Every 150 frames → **Every 300 frames** (~10 seconds)
- Scene change threshold: 15% → **25%** (less sensitive)

### 4. **Result**
Instead of announcing every small object detected, the system now:
- ✅ Only speaks about navigation-relevant obstacles
- ✅ Waits 15-20 seconds before repeating same alert
- ✅ Focuses on safety-critical items (vehicles, people, hazards)
- ✅ Gives periodic scene summaries every ~10 seconds
- ✅ More natural, GPS-like experience

## Example Output Comparison

### Before (Too Chatty):
```
[TTS] remote detected on right
[TTS] laptop detected on left  
[TTS] book detected on left
[TTS] tv detected on left
[TTS] book detected on center
[TTS] keyboard detected on right
[TTS] motorcycle detected on right
[TTS] person detected on center
[TTS] cell phone detected on center
[TTS] cat detected on center
[TTS] knife detected on right
...every second...
```

### After (Navigation-Friendly):
```
[TTS] person detected ahead
...15 seconds pass...
[TTS] car on right  
...15 seconds pass...
[Scene Description] "Living room with furniture"
...10 seconds pass...
[TTS] person approaching from left
```

## Configuration File
All settings in: `config.py`

Easy to adjust:
```python
SPATIAL_TRACKING["movement_threshold"] = 80.0  # Higher = less sensitive
SPATIAL_TRACKING["time_threshold"] = 15.0      # Longer = fewer repeats
SAFETY["alert_cooldown"] = 20.0                # Longer between same warnings
```

Human-friendly navigation mode activated! 🎯
