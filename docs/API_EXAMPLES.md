# API Usage Examples

## How to Use the Enhanced System

### 1. Basic Setup

```python
from inference.fusion_engine import FusionEngine
from audio.tts import TTSEngine

# Initialize TTS
tts = TTSEngine()

# Initialize Enhanced Fusion Engine
fusion = FusionEngine(
    tts_engine=tts,
    llm_backend="lm_studio",
    llm_url="http://localhost:1234/v1"
)
```

### 2. Update Spatial Context (Every Frame)

```python
import time

# Get detections from YOLO
detections = [
    {
        "label": "car",
        "confidence": 0.89,
        "bbox": [[320, 180, 640, 480]]  # [x1, y1, x2, y2]
    },
    {
        "label": "person",
        "confidence": 0.76,
        "bbox": [[100, 200, 250, 500]]
    }
]

# Update spatial tracking
fusion.update_spatial_context(
    detections=detections,
    timestamp=time.time(),
    frame_width=1280
)
```

**What happens:**
- Tracks each object with unique ID
- Detects new objects → announces "car detected on center"
- Tracks movement → announces "person moving left"
- Maintains spatial history for LLM

### 3. Process VLM Scene Description

```python
# VLM generates scene description
vlm_description = "Urban parking lot with several parked cars and pedestrians walking"

# Send to fusion engine
fusion.handle_vlm_description(vlm_description)
```

**What happens:**
- Stores description in context memory
- If user query is pending → triggers LLM to answer
- Otherwise → speaks description (if not redundant)

### 4. Handle User Query

```python
# User asks a question (from STT)
user_query = "Is it safe to cross?"

# Process query
fusion.handle_user_query(user_query)
```

**What happens:**
1. Stores query as pending
2. Speaks "Processing your question..."
3. Gets spatial context: `"car: center, close (tracked 15 frames)"`
4. Calls LLM with: Query + Spatial Context + Latest VLM Description
5. Speaks LLM answer: `"Caution: car at close range in center"`

### 5. Get Current Spatial State

```python
# Get brief summary
summary = fusion.get_spatial_summary()
print(summary)  # "car center, person left"

# Get full context for debugging
full_context = fusion.spatial_context.get_context_for_llm()
print(full_context)
```

**Output:**
```
=== CURRENT ENVIRONMENT ===
- car: center side, close (tracked for 15 frames)
- person: left side, medium distance (tracked for 8 frames)

=== RECENT EVENTS ===
- car appeared center at close distance
- person is approaching from center

=== SCENE UNDERSTANDING ===
- Urban parking lot with several parked cars
```

### 6. LLM Safety Analysis (Optional)

```python
# Analyze scene for unreported hazards
safety_alert = fusion.analyze_scene_safety()

if safety_alert:
    print(f"LLM detected hazard: {safety_alert}")
```

**What happens:**
- LLM analyzes spatial context + VLM description
- Returns warnings like: `"Approaching vehicle from left"`
- Auto-speaks if hazard detected

## Complete Integration Example

```python
import cv2
import time
from safety.frame_capture import Camera
from safety.yolo_detector import YoloDetector
from safety.safety_rules import SafetyRules
from audio.tts import TTSEngine
from inference.fusion_engine import FusionEngine
from reasoning.qwen_vlm import QwenVLM

# Initialize components
camera = Camera()
detector = YoloDetector()
safety = SafetyRules()
tts = TTSEngine()

# Enhanced fusion engine
fusion = FusionEngine(tts, llm_backend="lm_studio")

# VLM for scene understanding
qwen = QwenVLM(backend="lm_studio")

print("System started")

for frame in camera.stream():
    current_time = time.time()
    
    # 1. YOLO Detection
    detections = detector.detect(frame)
    
    # 2. Safety Check (CRITICAL)
    safety_result = safety.evaluate(detections)
    if safety_result:
        alert_type, message = safety_result
        fusion.handle_safety_alert(message, alert_type)
    
    # 3. Update Spatial Context
    fusion.update_spatial_context(detections, current_time, frame.shape[1])
    
    # 4. VLM Processing (every N frames)
    if should_run_vlm():  # Your sampling logic
        scene_desc = qwen.describe_scene(frame)
        fusion.handle_vlm_description(scene_desc)
    
    # 5. User Query (on 'L' key press)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('l'):
        query = get_user_query()  # Your STT function
        fusion.handle_user_query(query)
    
    # Display
    cv2.imshow("WalkSense", frame)
```

## Direct LLM Usage (Without Fusion)

```python
from inference.llm_reasoner import LLMReasoner

# Initialize LLM
llm = LLMReasoner(backend="lm_studio", api_url="http://localhost:1234/v1")

# Answer query
answer = llm.answer_query(
    user_query="What's on my left?",
    spatial_context="car: left side, close distance",
    scene_description="Urban street with parked vehicles"
)
print(answer)  # "A car is parked on your left at close range"

# Safety analysis
alert = llm.analyze_safety(
    spatial_context="car: center, close, approaching",
    scene_description="Busy intersection with moving traffic"
)
print(alert)  # "Caution: vehicle approaching from center"

# Navigation hint
hint = llm.generate_navigation_hint(
    spatial_context="person: left medium, chair: right far"
)
print(hint)  # "Path ahead is clear. Person on your left."
```

## Direct Spatial Context Manager Usage

```python
from inference.spatial_context_manager import SpatialContextManager
import time

# Initialize
context_mgr = SpatialContextManager(
    movement_threshold=30.0,  # pixels
    time_threshold=10.0       # seconds
)

# Update with detections
events = context_mgr.update(
    detections=[
        {"label": "car", "confidence": 0.89, "bbox": [[320, 180, 640, 480]]}
    ],
    timestamp=time.time(),
    frame_width=1280
)

# Check events
for event in events:
    if event['type'] == 'NEW_OBJECT':
        print(f"New {event['label']} at {event['direction']}, {event['distance']}")
    elif event['type'] == 'OBJECT_MOVED':
        print(f"{event['label']} moved to {event['direction']}")

# Add scene description
context_mgr.add_scene_description("Urban parking lot")

# Get formatted context
llm_context = context_mgr.get_context_for_llm()
print(llm_context)
```

## Configuration Options

### SpatialContextManager Tuning

```python
fusion = FusionEngine(tts)

# Adjust movement threshold (default: 30 pixels)
fusion.spatial_context.movement_threshold = 50.0  # Less sensitive

# Adjust time threshold (default: 10 seconds)
fusion.spatial_context.time_threshold = 15.0  # Longer cooldown

# Access tracker settings
fusion.spatial_context.tracker.iou_threshold = 0.4  # Higher = stricter matching
fusion.spatial_context.tracker.max_age = 20  # Keep tracks for 20 frames
```

### LLM Reasoner Backends

```python
# LM Studio (default)
fusion = FusionEngine(tts, llm_backend="lm_studio", llm_url="http://localhost:1234/v1")

# Ollama
fusion = FusionEngine(tts, llm_backend="ollama", llm_url="http://localhost:11434")

# Custom model
fusion.llm.model_name = "llama2-7b"
```

### Custom LLM Prompts

```python
# Modify system prompt
fusion.llm.system_prompt = """You are a navigation assistant for blind users.
Be extremely concise and safety-focused. Maximum 20 words per response."""

# Modify temperature for more/less creativity
# In llm_reasoner.py, edit _call_lm_studio() temperature parameter
```

## Error Handling

```python
try:
    fusion.handle_user_query("What's ahead?")
except Exception as e:
    print(f"LLM error: {e}")
    # Fallback to simple response
    tts.speak("I'm having trouble understanding the scene right now")
```

## Performance Monitoring

```python
import time

# Track spatial updates
start = time.time()
fusion.update_spatial_context(detections, time.time(), 1280)
print(f"Spatial update: {(time.time() - start) * 1000:.1f}ms")

# Track LLM response time
start = time.time()
answer = fusion.llm.answer_query(query, context, scene)
print(f"LLM query: {(time.time() - start):.2f}s")
```

Expected latencies:
- Spatial update: **1-5ms**
- LLM query: **1-3 seconds**
- Total overhead: **Negligible** (spatial tracking is very fast)
