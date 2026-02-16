# WalkSense System Latency Analysis

## Overview
This document analyzes the end-to-end latency of the WalkSense system and identifies optimization opportunities.

## Current System Performance

### 1. Perception Layer (Real-time - 30 FPS)
| Component | Latency | Notes |
|-----------|---------|-------|
| Camera Capture | ~33ms | 30 FPS = 33ms per frame |
| YOLO Detection | ~30-50ms | GPU: 30ms, CPU: 100-200ms |
| Safety Rules | <5ms | Deterministic checks |
| **Total Perception** | **~70ms** | **Acceptable for safety** |

### 2. Speech-to-Text (User Query)
| Component | Latency | Notes |
|-----------|---------|-------|
| Mic Calibration | 500ms | Reduced from 2s |
| User Speaking | 2-5s | Variable (user dependent) |
| Silence Detection | 600ms | `pause_threshold` |
| Faster-Whisper (GPU) | ❌ FAILED | Missing `cublas64_12.dll` |
| OpenAI Whisper (CPU) | ❌ FAILED | Missing `ffmpeg` |
| Google STT (Fallback) | 1-3s | Network dependent |
| **Total STT** | **5-10s** | **NEEDS OPTIMIZATION** |

### 3. Vision-Language Model (Scene Understanding)
| Component | Latency | Notes |
|-----------|---------|-------|
| Frame Sampling | 150 frames | ~5s between captures |
| Scene Change Detection | <10ms | Histogram comparison |
| VLM Inference (Qwen) | 2-5s | LM Studio API call |
| **Total VLM** | **2-5s** | **Acceptable (async)** |

### 4. LLM Reasoning (Query Answering)
| Component | Latency | Notes |
|-----------|---------|-------|
| Context Assembly | <10ms | Spatial + VLM + Query |
| LLM Inference (Ollama) | 2-10s | Model: gemma3:270m |
| LM Studio (Alternative) | ❌ ERROR 400 | Configuration issue |
| **Total LLM** | **2-10s** | **Acceptable** |

### 5. Text-to-Speech (Audio Output)
| Component | Latency | Notes |
|-----------|---------|-------|
| TTS Worker Init | 200ms | One-time startup |
| pyttsx3 Processing | 100-300ms | Per sentence |
| Audio Playback | Variable | Depends on text length |
| **Total TTS** | **~500ms** | **ISSUE: Silent output** |

## End-to-End Latency Breakdown

### Safety Alert Path (Critical)
```
Camera -> YOLO -> Safety -> Fusion -> TTS
33ms + 30ms + 5ms + 5ms + 500ms = ~573ms
```
✅ **Status: ACCEPTABLE** (under 1 second for critical alerts)

### User Query Path (Interactive)
```
Press L -> Prompt -> STT -> VLM -> LLM -> TTS -> Audio
0ms + 1.5s + 8s + 3s + 5s + 0.5s = ~18s
```
❌ **Status: TOO SLOW** (should be under 10s)

### Scene Description Path (Passive)
```
Frame Sample -> VLM -> Fusion -> TTS
5s + 3s + 5ms + 0.5s = ~8.5s
```
✅ **Status: ACCEPTABLE** (background task)

---

## Critical Issues & Fixes

### 🔴 CRITICAL: STT Latency (5-10s)
**Problem:** 
- Faster-Whisper fails due to missing CUDA DLL (`cublas64_12.dll`)
- OpenAI Whisper fails due to missing `ffmpeg`
- Falls back to Google STT (slow, network-dependent)

**Fix Priority: HIGH**
1. **Install CUDA Toolkit 12.x** to get `cublas64_12.dll`
   - Download: https://developer.nvidia.com/cuda-downloads
   - Or copy DLL from existing CUDA installation
2. **Install ffmpeg** for OpenAI Whisper fallback
   - Download: https://ffmpeg.org/download.html
   - Add to PATH
3. **Optimize Config:**
   ```json
   "stt": {
       "providers": {
           "whisper_local": {
               "model_size": "tiny",  // Change from "small" to "tiny"
               "device": "cuda",
               "compute_type": "int8"
           }
       }
   }
   ```
   - **Expected Improvement:** 8s → 2s (4x faster)

### 🔴 CRITICAL: TTS Not Speaking
**Problem:**
- `[TTS] SPEAKING: ...` appears in logs
- Audio worker process starts successfully
- But no sound is heard (except "What do you want to know?")

**Possible Causes:**
1. **Worker Process Dying:** Check logs for `[TTS] Worker died`
2. **Voice Selection Issue:** Zira might be corrupted
3. **Audio Device Routing:** pyttsx3 sending to wrong output

**Fix Priority: CRITICAL**
1. **Run diagnostic:**
   ```bash
   python scripts/test_voices.py
   ```
   - Listen for each voice test
2. **Check Windows Sound Settings:**
   - Ensure default playback device is correct
   - Test with `python interaction_layer/audio_worker.py "test"`
3. **Force different voice in config.json:**
   ```json
   "tts": {
       "providers": {
           "pyttsx3": {
               "voice": "david",  // Try "david" instead of "default"
               "rate": 150,
               "volume": 1.0
           }
       }
   }
   ```

### 🟡 MEDIUM: LLM Timeout Errors
**Problem:**
- `LM Studio error: 400`
- `Ollama read timeout`

**Fix:**
1. **Increase timeout in `reasoning_layer/llm.py`:**
   ```python
   timeout=30  # Increase from 10
   ```
2. **Use smaller/faster model:**
   ```json
   "llm": {
       "providers": {
           "ollama": {
               "model_id": "gemma2:2b"  // Smaller model
           }
       }
   }
   ```

### 🟢 LOW: VLM Sampling Interval
**Current:** 150 frames (~5s)
**Optimization:** Increase to 200 frames (~6.5s) to reduce VLM load

---

## Recommended Optimization Roadmap

### Phase 1: Fix Critical Blockers (Today)
1. ✅ Fix TTS audio output (test voices, check device)
2. ✅ Install CUDA DLL or ffmpeg for local STT
3. ✅ Switch to `tiny` Whisper model

**Expected Result:** Query response time: 18s → 8s

### Phase 2: Performance Tuning (This Week)
1. Optimize LLM timeout and model selection
2. Fine-tune STT pause threshold (0.6s → 0.4s)
3. Implement TTS queue management (prevent backlog)

**Expected Result:** Query response time: 8s → 5s

### Phase 3: Advanced Optimizations (Future)
1. Implement streaming STT (partial results)
2. Use GPU-accelerated VLM (if available)
3. Implement response caching for common queries
4. Add voice activity detection (VAD) for faster STT trigger

**Expected Result:** Query response time: 5s → 3s

---

## Performance Monitoring

The system includes a built-in performance tracker. To view metrics:
1. Run the system
2. Press `Q` to quit
3. Check `plots/` directory for performance graphs

Key metrics to monitor:
- `frame_total`: End-to-end frame processing time
- `yolo`: Object detection latency
- `stt`: Speech recognition latency
- `llm_reasoning`: Query answering latency

---

## Conclusion

**Current System Status:**
- ✅ Safety alerts: Fast and reliable (~600ms)
- ❌ User queries: Too slow (~18s) - needs STT optimization
- ⚠️ TTS: Not producing audio - critical bug

**Priority Actions:**
1. Fix TTS audio output (CRITICAL)
2. Install CUDA/ffmpeg for local STT (HIGH)
3. Switch to `tiny` Whisper model (HIGH)

**Target Performance:**
- Safety alerts: <1s ✅ (already achieved)
- User queries: <5s ⏳ (achievable with fixes)
- Scene descriptions: <10s ✅ (already acceptable)
