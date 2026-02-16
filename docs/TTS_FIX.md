# TTS Fix Summary

## Issues Identified

1. **Duplicate TTS Calls**: The main script (`run_enhanced_camera.py`) was calling `tts.speak(answer)` directly after the fusion engine already routed the answer through the router, which calls TTS. This caused the second call to clear the queue and potentially interrupt the first.

2. **Insufficient Logging**: TTS operations were logged at DEBUG level, making it hard to diagnose issues in production.

3. **No Router Visibility**: The router wasn't logging when it was routing messages to TTS, making it difficult to trace the flow.

## Changes Made

### 1. `scripts/run_enhanced_camera.py`
- **Removed** redundant `tts.speak(answer)` call on line 430
- The router already handles speaking the answer when routing RESPONSE events
- Added comment explaining why the direct TTS call was removed

### 2. `interaction_layer/tts.py`
- **Enhanced logging** from DEBUG to INFO level for TTS operations
- Added emojis (🔊 and ✓) to make TTS events more visible in logs
- Now logs the full text being spoken, not just the first 50 characters

### 3. `fusion_layer/router.py`
- **Added logging** for RESPONSE event routing
- **Added logging** for SCENE_DESC event routing (both when spoken and when suppressed)
- Helps track the complete flow from event → router → TTS

## How TTS Works Now

The correct flow is:

```
Event Generated → FusionEngine → Router → TTS Engine → Speaker
```

### Event Types and TTS Behavior:

1. **CRITICAL_ALERT**: Always speaks (overrides mute)
   - Example: "Danger! Car approaching"

2. **WARNING**: Speaks if not redundant and not muted
   - Example: "Warning! Person close to your left"

3. **RESPONSE**: Speaks if not muted
   - Example: "Checking on: what obstacles are in front of me?"
   - Example: "AI: There is a chair 2 meters ahead"

4. **INFO**: Speaks if not redundant and not muted
   - Example: Basic object detection alerts

5. **SCENE_DESC**: Speaks if not redundant and not muted
   - Example: "The scene shows a hallway with a door on the left"

## Testing

### Quick Test
Run the test script to verify basic TTS functionality:
```bash
python test_tts.py
```

### Full System Test
1. Start the enhanced camera:
   ```bash
   python -m scripts.run_enhanced_camera
   ```

2. Press `S` to start the system
   - Should hear: "System Started"

3. Press `L` to ask a question
   - Should hear: "What do you want to know?"
   - Speak your question
   - Should hear: "Checking on: [your question]"
   - After VLM processes: Should hear the AI's answer

4. Press `M` to toggle mute
   - Should hear: "Audio Muted" or "Audio Active"

5. Watch the logs for TTS activity:
   - Look for `[TTS] 🔊 Speaking:` messages
   - Look for `[ROUTER] Routing RESPONSE:` messages
   - Look for `[TTS] ✓ Finished speaking` messages

## Debugging

If TTS still doesn't work:

1. **Check logs** for TTS worker initialization:
   - Should see: `[TTS] Worker thread started successfully`

2. **Check if messages are being routed**:
   - Look for `[ROUTER] Routing RESPONSE:` or `[ROUTER] Routing SCENE_DESC:`

3. **Check if TTS is receiving messages**:
   - Look for `[TTS] 🔊 Speaking:`

4. **Check if muted**:
   - Press `M` to toggle mute state

5. **Check pyttsx3 installation**:
   ```bash
   pip install --upgrade pyttsx3
   ```

6. **Check Windows audio**:
   - Ensure volume is up
   - Ensure correct audio output device is selected
   - Test with: `python test_tts.py`

## Configuration

TTS settings are in `config.json`:

```json
"tts": {
    "active_provider": "pyttsx3",
    "providers": {
        "pyttsx3": {
            "voice": "default",
            "rate": 150,
            "volume": 1.0
        }
    }
}
```

- **voice**: "default" uses Zira (preferred), or specify a voice name
- **rate**: Speech rate (words per minute), default 150
- **volume**: 0.0 to 1.0, default 1.0
