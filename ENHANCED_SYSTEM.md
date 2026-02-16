# Enhanced WalkSense: Spatial-Temporal Context + LLM Reasoning

## What's New

The enhanced WalkSense system adds **spatial-temporal object tracking** and **LLM-based query answering** to provide more intelligent and context-aware assistance.

### New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENHANCED FUSION ENGINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │ YOLO Detector│───▶│  SpatialContextManager           │  │
│  │  (Every Frame)│    │  - IoU Object Tracking           │  │
│  └──────────────┘    │  - Position History              │  │
│                       │  - Velocity Estimation           │  │
│                       │  - Movement Detection            │  │
│  ┌──────────────┐    └──────────────┬───────────────────┘  │
│  │ Qwen VLM     │                   │                       │
│  │ (Scene Desc) │────┐              │                       │
│  └──────────────┘    │              │                       │
│                       │              │                       │
│  ┌──────────────┐    │    ┌─────────▼─────────┐            │
│  │ User Query   │────┴───▶│   LLM Reasoner     │            │
│  │ (STT)        │         │  - Query Answering  │            │
│  └──────────────┘         │  - Safety Analysis  │            │
│                            │  - Context Fusion   │            │
│                            └─────────┬───────────┘            │
│                                      │                        │
│                            ┌─────────▼───────────┐           │
│                            │  DecisionRouter     │           │
│                            │  - TTS              │           │
│                            │  - Haptics          │           │
│                            │  - Buzzer           │           │
│                            └─────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. SpatialContextManager
**File:** `inference/spatial_context_manager.py`

**Features:**
- **IoU-based Object Tracking**: Maintains object identity across frames
- **Spatial Awareness**: Tracks object position (left/center/right, distance estimation)
- **Temporal Tracking**: Monitors object persistence and movement
- **Velocity Estimation**: Detects approaching objects
- **Structured Context**: Provides formatted context for LLM reasoning

**Data Structure:**
```python
@dataclass
class ObjectState:
    track_id: int          # Unique tracking ID
    label: str            # Object class
    bbox: List[float]     # Bounding box
    center: np.ndarray    # Center position
    direction: str        # "left", "center", "right"
    distance: str         # "close", "medium", "far"
    velocity: np.ndarray  # Movement vector
    first_seen: float     # First detection timestamp
    frames_tracked: int   # Persistence count
```

**Example Context Output:**
```
=== CURRENT ENVIRONMENT ===
- car: left side, close (tracked for 15 frames)
- person: center, medium distance (tracked for 8 frames)

=== RECENT EVENTS ===
- car appeared left at close distance
- person is approaching from center

=== SCENE UNDERSTANDING ===
- Busy urban street with pedestrians and vehicles
```

### 2. LLMReasoner
**File:** `inference/llm_reasoner.py`

**Capabilities:**
1. **Query Answering**: Combines spatial context + VLM description to answer user questions
2. **Safety Analysis**: Detects unreported hazards using LLM reasoning
3. **Navigation Hints**: Generates proactive guidance

**Supported Backends:**
- LM Studio (default)
- Ollama
- OpenAI API (extensible)

**Example Query Flow:**
```
User: "What's in front of me?"

LLM Input:
  Spatial Context: "car: center, close (tracked 12 frames)"
  Scene Description: "Urban parking lot with cars and pedestrians"
  Query: "What's in front of me?"

LLM Output: "A car is directly ahead at close range. Proceed with caution."
```

### 3. Enhanced FusionEngine
**File:** `inference/fusion_engine.py`

**New Methods:**

```python
# Update spatial tracking with new detections
fusion.update_spatial_context(detections, timestamp, frame_width)

# Process VLM description (triggers LLM if query pending)
fusion.handle_vlm_description(scene_description)

# Handle user query (answered by LLM with full context)
fusion.handle_user_query(user_query)

# Get current spatial state summary
summary = fusion.get_spatial_summary()  # "car left, person center"

# LLM safety analysis
alert = fusion.analyze_scene_safety()
```

## Usage

### Run Enhanced Demo

```bash
python scripts/run_enhanced_camera.py
```

### Configuration

Edit configuration at top of `run_enhanced_camera.py`:

```python
# VLM Backend
QWEN_BACKEND = "lm_studio"  # or "huggingface"
LM_STUDIO_URL = "http://localhost:1234/v1"

# LLM Backend (for query answering)
LLM_BACKEND = "lm_studio"  # or "ollama"
LLM_URL = "http://localhost:1234/v1"
```

### Controls

| Key | Action |
|-----|--------|
| `S` | Start system |
| `L` | Ask question (push-to-talk) |
| `M` | Toggle mute |
| `Q` | Quit |

## Data Flow Example

### 1. Object Detection → Spatial Tracking

```python
# Frame 1: New car detected
[YOLO] car detected at (320, 180, 640, 480), conf=0.89
[SpatialContext] NEW_OBJECT: track_id=42, car, center, close
[TTS] "car detected on center"

# Frame 15: Same car moved
[SpatialContext] OBJECT_MOVED: track_id=42, movement=45px
[TTS] "car moving center"

# Frame 30: Car disappeared
[SpatialContext] Track 42 removed (disappeared)
```

### 2. User Query → LLM Reasoning

```python
# User presses 'L' and speaks
[STT] "Is it safe to cross?"

# FusionEngine receives query
[FusionEngine] Pending query stored
[TTS] "Processing your question..."

# Next VLM result arrives
[QWEN] "Pedestrian crossing with stopped vehicles"
[FusionEngine] VLM description received, query pending

# LLM Reasoning
[FusionEngine] Answering query: "Is it safe to cross?"
[LLM] Input Context:
  - car: left side, close (tracked 42 frames)
  - person: center, medium (tracked 12 frames)
  - VLM: "Pedestrian crossing with stopped vehicles"

[LLM] Output: "Yes, vehicles appear stopped. Check left for approaching cars."
[TTS] "Yes, vehicles appear stopped. Check left for approaching cars."
```

### 3. Spatial Context Memory

The system maintains:
- **Active objects** (currently visible)
- **Spatial history** (last 20 events): "car appeared left", "person approaching"
- **Scene memory** (last 5 VLM descriptions)
- **Last positions** (for movement detection)

## Advantages Over String-Based Context

| Feature | Old ContextManager | New SpatialContextManager |
|---------|-------------------|---------------------------|
| Redundancy Detection | Text similarity only | Spatial + temporal state |
| Object Identity | None (each frame independent) | IoU tracking maintains ID |
| Movement Awareness | None | Velocity + position deltas |
| Spatial Understanding | None | Direction + distance estimation |
| LLM Integration | Not designed | Structured context format |

**Example:**
```python
# OLD: Would announce "car detected" every frame
"car detected"
"car detected"  # Redundant? (60% similar → suppressed)

# NEW: Tracks same car, only announces state changes
"car appeared center at close distance"  # Frame 1
# ... silence for 10 seconds ...
"car is approaching from center"  # Only if moved significantly
```

## Tuning Parameters

### SpatialContextManager

```python
SpatialContextManager(
    movement_threshold=30.0,  # Pixels - minimum movement to announce
    time_threshold=10.0,      # Seconds - cooldown between announcements
    max_history=20            # Events to keep in memory
)
```

### IOUTracker

```python
IOUTracker(
    iou_threshold=0.3,  # Minimum IoU for track matching
    max_age=30          # Frames - drop track if not seen
)
```

### LLMReasoner

```python
# In llm_reasoner.py, adjust prompts:
system_prompt = "You are an AI assistant..."  # Modify personality
max_tokens = 100  # Response length limit
temperature = 0.7  # Randomness (0.0 = deterministic, 1.0 = creative)
```

## Future Enhancements

1. **Depth Estimation**: Use monocular depth to get accurate distance
2. **Trajectory Prediction**: Predict where objects will be in 2-3 seconds
3. **Semantic Scene Graphs**: Relationships like "person NEAR car"
4. **Memory Persistence**: Save known locations/landmarks
5. **Emotion-Aware TTS**: Adjust voice based on urgency

## Testing

Run the enhanced system and test these scenarios:

1. **Stationary object**: Should announce once, then silence until moved
2. **Approaching object**: Should detect velocity and warn
3. **User query**: "What's on my left?" → Should use spatial context
4. **Safety analysis**: LLM should detect hazards VLM missed

## Troubleshooting

**LLM not responding:**
- Check that LM Studio or Ollama is running
- Verify correct API URL in configuration
- Check console for LLM error messages

**Objects not tracked:**
- Lower `iou_threshold` if tracks lost too quickly
- Increase `max_age` for objects that disappear temporarily

**Too many announcements:**
- Increase `movement_threshold` to only announce larger movements
- Increase `time_threshold` for longer cooldowns

## Performance

Approximate latencies on mid-range CPU:
- YOLO inference: 30-50ms
- IoU tracking: 1-5ms (negligible)
- VLM inference: 2-5s (async, non-blocking)
- LLM query answer: 1-3s
- Total perception loop: 30-50ms (real-time capable)
