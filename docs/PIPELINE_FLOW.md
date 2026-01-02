# Enhanced WalkSense Data Flow

## Complete Pipeline Visualization

```
┌─────────────┐
│   CAMERA    │ 30 FPS @ 1280x720
└──────┬──────┘
       │
       ├──────────────────────────────────────────────────────────┐
       │                                                            │
       ▼                                                            ▼
┌────────────────┐                                      ┌──────────────────┐
│ YOLO Detector  │ EVERY FRAME (SAFETY CRITICAL)        │  Frame Sampler   │
│   yolov8n.pt   │────────┐                             │  + Scene Detect  │
└────────────────┘        │                             └────────┬─────────┘
                          │                                      │
                          │                          Every 150 frames + 15% scene change
                          │                                      │
                          │                                      ▼
                          │                             ┌──────────────────┐
                          │                             │   Qwen VLM       │
                          │                             │  (Async Thread)  │
                          │                             └────────┬─────────┘
                          │                                      │
                          │                                      │ Scene Description
                          │                                      │ (2-5 seconds)
                          │                                      │
                          ▼                                      │
                 ┌──────────────────────────────────────────────┴──────┐
                 │           ENHANCED FUSION ENGINE                    │
                 │                                                     │
                 │  ┌──────────────────────────────────────────────┐  │
                 │  │   SpatialContextManager                      │  │
                 │  │                                              │  │
  Detections ───────▶ IoU Tracker                                 │  │
  [list]           │  │  ├─ Track car (ID: 42)                    │  │
                 │  │  │   └─ Position: center, close             │  │
                 │  │  │       Velocity: approaching              │  │
                 │  │  │       Frames: 15                         │  │
                 │  │  │                                           │  │
                 │  │  ├─ Track person (ID: 73)                   │  │
                 │  │  │   └─ Position: left, medium              │  │
                 │  │  │                                           │  │
                 │  │  └─ EVENTS:                                 │  │
                 │  │      • NEW_OBJECT                           │  │
                 │  │      • OBJECT_MOVED                         │  │
                 │  │                                              │  │
                 │  │  Spatial History:                           │  │
                 │  │  "car appeared center at close distance"    │  │
                 │  │  "person is approaching from center"        │  │
                 │  │                                              │  │
                 │  │  Scene Memory (last 5):                     │  │
                 │  │  "Urban parking lot with stopped cars"      │  │
Scene Desc ──────────▶│  "Pedestrian crossing ahead"              │  │
  (string)       │  └──────────────┬───────────────────────────────┘  │
                 │                 │                                  │
                 │                 │ get_context_for_llm()            │
                 │                 │                                  │
User Query ──────────────────┐     │                                  │
  "Is it safe?"  │           │     │                                  │
                 │           │     ▼                                  │
                 │           │  ┌──────────────────────────────────┐ │
                 │           └─▶│       LLM Reasoner               │ │
                 │              │  (LM Studio / Ollama)            │ │
                 │              │                                  │ │
                 │              │  Input:                          │ │
                 │              │  ┌─────────────────────────────┐ │ │
                 │              │  │ === CURRENT ENVIRONMENT === │ │ │
                 │              │  │ - car: center, close        │ │ │
                 │              │  │ - person: left, medium      │ │ │
                 │              │  │                             │ │ │
                 │              │  │ === RECENT EVENTS ===       │ │ │
                 │              │  │ - car appeared center       │ │ │
                 │              │  │                             │ │ │
                 │              │  │ === SCENE UNDERSTANDING === │ │ │
                 │              │  │ - Pedestrian crossing ahead │ │ │
                 │              │  │                             │ │ │
                 │              │  │ USER QUESTION:              │ │ │
                 │              │  │ "Is it safe?"               │ │ │
                 │              │  └─────────────────────────────┘ │ │
                 │              │                                  │ │
                 │              │  Output (1-3s):                  │ │
                 │              │  "Yes, vehicles stopped.         │ │
                 │              │   Check left for approaching."   │ │
                 │              └──────────────┬───────────────────┘ │
                 │                             │                     │
                 │                             │ Answer               │
                 │                             │                     │
                 │              ┌──────────────▼───────────────────┐ │
                 │              │     DecisionRouter               │ │
                 │              │                                  │ │
                 │              │  Priority Queue:                 │ │
                 │              │  1. CRITICAL_ALERT (override mute)│ │
                 │              │  2. WARNING                      │ │
                 │              │  3. RESPONSE (user query answer) │ │
                 │              │  4. SCENE_DESC                   │ │
                 │              │                                  │ │
                 │              │  Redundancy Check:               │ │
                 │              │  SequenceMatcher + Silence Window│ │
                 │              └──────────────┬───────────────────┘ │
                 └─────────────────────────────┼─────────────────────┘
                                               │
                          ┌────────────────────┼────────────────┐
                          │                    │                │
                          ▼                    ▼                ▼
                   ┌────────────┐      ┌────────────┐   ┌────────────┐
                   │    TTS     │      │  Haptics   │   │   Buzzer   │
                   │  pyttsx3   │      │  (vibrate) │   │   (beep)   │
                   └────────────┘      └────────────┘   └────────────┘
                          │
                          ▼
                   ┌────────────┐
                   │   AUDIO    │
                   │   OUTPUT   │
                   └────────────┘
```

## Key Differences from Old System

### Old Pipeline (String-Based)
```
YOLO → Safety Rules → TTS
VLM → String Similarity Check → TTS (if not redundant)
User Query → Hardcoded Response → TTS
```

### New Pipeline (Spatial-Temporal + LLM)
```
YOLO → IoU Tracker → Spatial Events → TTS
VLM → Context Memory → LLM Reasoning → TTS
User Query + Spatial Context + VLM Description → LLM → TTS
```

## Example Execution Timeline

```
Time | Component         | Action
-----|-------------------|--------------------------------------------------
0.0s | Camera            | Frame 1 captured
0.03s| YOLO              | Detections: [car @(320,180,640,480)]
0.04s| SpatialContext    | NEW track_id=42: car, center, close
0.04s| SafetyRules       | No critical hazards
0.05s| DecisionRouter    | Route event: "car detected on center"
0.06s| TTS               | Speak: "car detected on center"
     |                   |
3.0s | Camera            | Frame 90 captured (sampler triggers)
3.03s| YOLO              | Detections: [car @(325,200,645,500)]
3.04s| SpatialContext    | UPDATE track_id=42: moved 35px, velocity=+3px/s
3.04s| SceneDetector     | Scene changed: 18% (above threshold)
3.04s| QwenWorker        | Async VLM inference started
     |                   |
5.2s | QwenWorker        | VLM result: "Urban parking lot with cars"
5.21s| FusionEngine      | handle_vlm_description() called
5.21s| SpatialContext    | Scene memory updated
5.22s| DecisionRouter    | Route event: "Urban parking lot..."
5.23s| TTS               | Speak: "Urban parking lot with cars"
     |                   |
8.0s | User              | Presses 'L' key
8.1s | STT               | Listening... "Is it safe to cross?"
8.5s | FusionEngine      | handle_user_query() called
8.51s| TTS               | Speak: "Processing your question..."
8.52s| LLM               | API call started
     |                   | Context: "car: center, close (42 frames)"
     |                   | Scene: "Urban parking lot with cars"
     |                   | Query: "Is it safe to cross?"
10.3s| LLM               | Response: "Caution: car center at close range"
10.31s| FusionEngine     | _answer_user_query() completes
10.32s| TTS              | Speak: "Caution: car center at close range"
```

## Memory Footprint

| Component | Storage | Size |
|-----------|---------|------|
| Active tracks (10 objects) | Dict[int, ObjectState] | ~2 KB |
| Spatial history (20 events) | deque[str] | ~1 KB |
| Scene memory (5 descriptions) | deque[dict] | ~1 KB |
| Last positions (10 objects) | Dict[int, np.array] | ~500 bytes |
| **Total** | | **~5 KB** |

Compared to VLM (2GB VRAM) and YOLO (40MB), context is negligible.
