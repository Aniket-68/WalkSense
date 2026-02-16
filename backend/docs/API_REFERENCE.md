# WalkSense API Reference

## Table of Contents

1. [Perception Layer](#perception-layer)
2. [Reasoning Layer](#reasoning-layer)
3. [Fusion Layer](#fusion-layer)
4. [Interaction Layer](#interaction-layer)
5. [Infrastructure](#infrastructure)

---

## Perception Layer

### Camera (`perception_layer/camera.py`)

#### Class: `Camera`

```python
class Camera:
    """
    Video capture interface supporting hardware cameras and video file simulation.
    
    Attributes:
        cap: cv2.VideoCapture instance
        source_type: str - 'hardware' or 'simulation'
        loop: bool - Whether to loop video in simulation mode
    """
```

**Methods**:

```python
def __init__(self, cam_id: Optional[int] = None, 
             width: Optional[int] = None, 
             height: Optional[int] = None) -> None:
    """
    Initialize camera capture device.
    
    Args:
        cam_id: Camera device index (None = use config)
        width: Frame width in pixels (None = use config)
        height: Frame height in pixels (None = use config)
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
```

```python
def stream(self) -> Generator[np.ndarray, None, None]:
    """
    Generate video frames continuously.
    
    Yields:
        np.ndarray: BGR frame of shape (height, width, 3)
        
    Example:
        >>> camera = Camera()
        >>> for frame in camera.stream():
        ...     process(frame)
    """
```

```python
def release(self) -> None:
    """Release camera resources."""
```

---

### YOLO Detector (`perception_layer/detector.py`)

#### Class: `YoloDetector`

```python
class YoloDetector:
    """
    YOLO-based object detection with configurable models and devices.
    
    Attributes:
        model: YOLO - Ultralytics YOLO model instance
        device: str - 'cuda' or 'cpu'
        conf_threshold: float - Minimum confidence for detections
    """
```

**Methods**:

```python
def __init__(self, model_path: Optional[str] = None) -> None:
    """
    Initialize YOLO detector.
    
    Args:
        model_path: Path to YOLO weights (None = use config)
        
    Note:
        Automatically falls back to CPU if CUDA unavailable
    """
```

```python
def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Perform object detection on a frame.
    
    Args:
        frame: Input image (BGR format)
        
    Returns:
        List of detection dictionaries:
        [
            {
                "label": str,           # Object class name
                "confidence": float,    # Detection confidence (0-1)
                "bbox": [[x1, y1, x2, y2]]  # Bounding box coords
            },
            ...
        ]
        
    Example:
        >>> detector = YoloDetector()
        >>> detections = detector.detect(frame)
        >>> print(detections[0]['label'])
        'person'
    """
```

---

### Safety Rules (`perception_layer/rules.py`)

#### Class: `SafetyRules`

```python
class SafetyRules:
    """
    Deterministic safety classification based on detected objects.
    
    Class Attributes:
        CRITICAL_OBJECTS: Set of hazardous object classes
        WARNING_OBJECTS: Set of cautionary object classes
        INFO_OBJECTS: Set of informational object classes
        THRESHOLDS: Dict of confidence thresholds per severity
    """
```

**Methods**:

```python
def evaluate(self, detections: List[Dict]) -> Optional[Tuple[str, str]]:
    """
    Evaluate detections for safety hazards.
    
    Args:
        detections: List of detection dictionaries from YoloDetector
        
    Returns:
        Tuple of (severity, message) or None:
        - ("CRITICAL_ALERT", "Danger! car detected ahead...")
        - ("WARNING", "Warning! pole ahead...")
        - ("INFO", "chair nearby")
        - None (no significant objects)
        
    Example:
        >>> rules = SafetyRules()
        >>> result = rules.evaluate(detections)
        >>> if result:
        ...     severity, message = result
        ...     print(f"{severity}: {message}")
    """
```

---

### Alert Events (`perception_layer/alerts.py`)

#### Class: `AlertEvent`

```python
@dataclass
class AlertEvent:
    """
    Immutable event representing a system alert.
    
    Attributes:
        type: str - Severity level (CRITICAL_ALERT, WARNING, INFO, RESPONSE, SCENE_DESC)
        message: str - Human-readable alert message
        timestamp: float - Unix timestamp of event creation
    """
    type: str
    message: str
    timestamp: float = field(default_factory=time.time)
```

---

## Reasoning Layer

### Vision-Language Model (`reasoning_layer/vlm.py`)

#### Class: `QwenVLM`

```python
class QwenVLM:
    """
    Vision-language model wrapper supporting multiple backends.
    
    Backends:
        - lm_studio: LM Studio API server
        - huggingface: Local Transformers model
        - ollama: Ollama API server
    
    Attributes:
        backend: str - Active backend name
        url: str - API endpoint URL (if applicable)
        model_id: str - Model identifier
    """
```

**Methods**:

```python
def __init__(self, 
             backend: Optional[str] = None,
             model_id: Optional[str] = None,
             lm_studio_url: Optional[str] = None) -> None:
    """
    Initialize VLM with specified backend.
    
    Args:
        backend: Backend provider (None = use config)
        model_id: Model identifier (None = use config)
        lm_studio_url: LM Studio URL override
        
    Raises:
        ValueError: If backend is unknown
    """
```

```python
def describe_scene(self, 
                   frame: np.ndarray, 
                   context: str = "") -> str:
    """
    Generate natural language scene description.
    
    Args:
        frame: Input image (BGR format)
        context: Additional context string (object detections, user query)
        
    Returns:
        Natural language scene description
        
    Example:
        >>> vlm = QwenVLM()
        >>> desc = vlm.describe_scene(frame, "USER QUESTION: What's ahead?")
        >>> print(desc)
        "A person standing 2 meters ahead with a bag"
        
    Note:
        Priority given to user questions if context contains "USER QUESTION:"
    """
```

---

### Language Model Reasoner (`reasoning_layer/llm.py`)

#### Class: `LLMReasoner`

```python
class LLMReasoner:
    """
    LLM-based reasoning for query answering and safety analysis.
    
    Backends:
        - lm_studio: LM Studio API
        - ollama: Ollama API
        - openai: OpenAI API (future)
    
    Attributes:
        backend: str - Active backend
        api_url: str - API endpoint
        model_name: str - Model identifier
        system_prompt: str - Jarvis-style system instructions
    """
```

**Methods**:

```python
def __init__(self,
             backend: str = "lm_studio",
             api_url: str = "http://localhost:1234/v1",
             model_name: str = "phi-4") -> None:
    """Initialize LLM reasoner with backend configuration."""
```

```python
def check_health(self) -> bool:
    """
    Verify backend connection.
    
    Returns:
        True if backend is responsive, False otherwise
    """
```

```python
def answer_query(self,
                 user_query: str,
                 spatial_context: str,
                 scene_description: Optional[str] = None) -> str:
    """
    Answer user query using spatial and visual context.
    
    Args:
        user_query: User's question from STT
        spatial_context: Spatial tracking information
        scene_description: Latest VLM scene description
        
    Returns:
        Concise answer (≤25 words) grounded in visual evidence
        
    Example:
        >>> llm = LLMReasoner()
        >>> answer = llm.answer_query(
        ...     "Is there a chair?",
        ...     "chair: center, close",
        ...     "A brown chair in the middle of the room"
        ... )
        >>> print(answer)
        "Yes, there's a brown chair directly in front of you"
    """
```

---

## Fusion Layer

### Fusion Engine (`fusion_layer/engine.py`)

#### Class: `FusionEngine`

```python
class FusionEngine:
    """
    Central orchestrator coordinating perception, reasoning, and interaction.
    
    Attributes:
        router: DecisionRouter - Alert routing logic
        runtime: RuntimeState - System state management
        llm: LLMReasoner - Language model interface
        spatial: SpatialContextManager - Object tracking
        pending_query: Optional[str] - Current user query awaiting response
    """
```

**Methods**:

```python
def __init__(self,
             tts_engine: TTSEngine,
             llm_backend: str = "lm_studio",
             llm_url: str = "http://localhost:1234/v1",
             llm_model: str = "qwen/qwen3-vl-4b") -> None:
    """Initialize fusion engine with TTS and LLM configuration."""
```

```python
def handle_safety_alert(self,
                        message: str,
                        alert_type: str = "CRITICAL_ALERT") -> None:
    """
    Process immediate safety hazard.
    
    Args:
        message: Alert description
        alert_type: Severity level
        
    Note:
        Includes automatic cooldown to prevent spam
    """
```

```python
def handle_user_query(self, query: str) -> None:
    """
    Process user voice query.
    
    Args:
        query: Transcribed user question
        
    Note:
        Sends immediate acknowledgment, then waits for next VLM frame
    """
```

```python
def handle_vlm_description(self, text: str) -> str:
    """
    Process VLM scene description.
    
    Args:
        text: Scene description from VLM
        
    Returns:
        LLM-generated answer if query pending, else original text
        
    Side Effects:
        - Clears pending_query if answered
        - Updates spatial scene memory
        - Routes response through DecisionRouter
    """
```

```python
def update_spatial_context(self,
                           detections: List[Dict],
                           timestamp: float,
                           frame_width: int = 1280) -> None:
    """
    Update object tracking and spatial awareness.
    
    Args:
        detections: Current frame detections
        timestamp: Current time (Unix)
        frame_width: Frame width for direction calculation
    """
```

```python
def get_spatial_summary(self) -> str:
    """
    Get brief spatial state summary for UI.
    
    Returns:
        Comma-separated list of tracked objects (e.g., "person left, chair center")
    """
```

---

### Spatial Context Manager (`fusion_layer/context.py`)

#### Class: `SpatialContextManager`

```python
class SpatialContextManager:
    """
    Tracks objects across frames with spatial-temporal awareness.
    
    Features:
        - IoU-based object tracking
        - Direction estimation (left/center/right)
        - Distance estimation (proximity-based)
        - Movement detection
        - Scene memory for LLM context
    
    Attributes:
        tracker: IOUTracker - Object persistence tracker
        active_objects: Dict[int, ObjectState] - Currently tracked objects
        spatial_history: deque - Recent spatial events
        scene_memory: deque - Recent VLM descriptions
    """
```

**Methods**:

```python
def update(self,
           detections: List[Dict],
           timestamp: float,
           frame_width: int = 1280) -> List[Dict]:
    """
    Update tracking with new detections.
    
    Args:
        detections: Current frame detections
        timestamp: Current Unix timestamp
        frame_width: Frame width for direction calculation
        
    Returns:
        List of spatial events:
        [
            {
                "type": "NEW_OBJECT" | "OBJECT_MOVED",
                "track_id": int,
                "label": str,
                "direction": "left" | "center" | "right",
                "distance": "very close" | "close" | "medium distance" | "far",
                ...
            },
            ...
        ]
    """
```

```python
def get_context_for_llm(self) -> str:
    """
    Generate formatted context for LLM reasoning.
    
    Returns:
        Multi-line string containing:
        - Current environment (active objects)
        - Recent events (spatial history)
        - Scene understanding (VLM descriptions)
        
    Example:
        === CURRENT ENVIRONMENT ===
        - person: center side, close (tracked for 45 frames)
        
        === RECENT EVENTS ===
        - person appeared center at close distance
        
        === SCENE UNDERSTANDING ===
        - A person wearing blue shirt standing ahead
    """
```

---

### Decision Router (`fusion_layer/router.py`)

#### Class: `DecisionRouter`

```python
class DecisionRouter:
    """
    Routes alerts to appropriate output channels (TTS, haptics, etc.).
    
    Priority Levels:
        1. CRITICAL_ALERT: Overrides mute, strong haptic + buzzer
        2. WARNING: Medium feedback if not redundant
        3. RESPONSE/INFO: Confirmation haptic
        4. SCENE_DESC: Passive description (redundancy filtered)
    
    Attributes:
        tts: TTSEngine - Text-to-speech interface
        context_manager: ContextManager - Redundancy filter
        aux: AuxController - Haptic/buzzer/LED controller
        muted: bool - Audio mute state
    """
```

**Methods**:

```python
def route(self, event: AlertEvent) -> None:
    """
    Route event to appropriate output channels.
    
    Args:
        event: AlertEvent to process
        
    Logic:
        - CRITICAL_ALERT: Always spoken + strong haptic/buzzer
        - WARNING: Spoken if not redundant + medium haptic
        - RESPONSE: Always spoken + pulse haptic
        - SCENE_DESC: Spoken if not redundant (passive)
    """
```

```python
def toggle_mute(self) -> bool:
    """
    Toggle audio mute state.
    
    Returns:
        New mute state (True = muted)
        
    Note:
        CRITICAL_ALERT always overrides mute
    """
```

---

## Interaction Layer

### Speech-to-Text (`interaction_layer/stt.py`)

#### Class: `STTListener`

```python
class STTListener:
    """
    Speech recognition with multiple backend support.
    
    Providers:
        - google: Google Speech Recognition API
        - whisper_local: faster-whisper (GPU-accelerated)
        - whisper_api: OpenAI Whisper API
    
    Attributes:
        recognizer: sr.Recognizer
        mic: sr.Microphone
        device_index: Optional[int] - Microphone device ID
    """
```

**Methods**:

```python
def listen_once(self, timeout: Optional[int] = None) -> Optional[str]:
    """
    Capture and transcribe single voice input.
    
    Args:
        timeout: Maximum wait time for speech start (None = use config)
        
    Returns:
        Transcribed text or None if timeout/error
        
    Example:
        >>> listener = STTListener()
        >>> text = listener.listen_once()
        >>> print(f"User said: {text}")
        
    Note:
        Automatically calibrates for ambient noise before listening
    """
```

---

### Text-to-Speech (`interaction_layer/tts.py`)

#### Class: `TTSEngine`

```python
class TTSEngine:
    """
    Text-to-speech audio output.
    
    Uses subprocess to run audio_worker.py for non-blocking speech.
    
    Attributes:
        script_path: str - Path to audio_worker.py
        current_process: Optional[subprocess.Popen]
    """
```

**Methods**:

```python
def speak(self, text: str) -> None:
    """
    Speak text aloud (interrupts previous speech).
    
    Args:
        text: Text to synthesize
        
    Side Effects:
        - Kills any existing speech process
        - Logs speech to performance.log
        - Spawns audio_worker.py subprocess
    """
```

```python
def stop(self) -> None:
    """Immediately stop current speech playback."""
```

---

## Infrastructure

### Config Loader (`infrastructure/config.py`)

#### Class: `Config`

```python
class Config:
    """
    Centralized configuration management.
    
    Loads settings from config.json and provides dot-notation access.
    
    Class Attributes:
        _config_path: str - Path to config.json
        _data: Dict - Loaded configuration data
    """
```

**Methods**:

```python
@classmethod
def get(cls, key_path: str, default: Any = None) -> Any:
    """
    Retrieve configuration value using dot-separated path.
    
    Args:
        key_path: Dot-separated key path (e.g., "vlm.active_provider")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> device = Config.get("detector.device", "cpu")
        >>> print(device)
        'cuda'
    """
```

---

### Performance Tracker (`infrastructure/performance.py`)

#### Class: `PerformanceTracker`

```python
class PerformanceTracker:
    """
    Latency tracking and performance visualization.
    
    Features:
        - Named timers for components
        - Automatic statistics (avg, min, max)
        - Chart generation
        - Filtered terminal logging
    
    Attributes:
        metrics: defaultdict[str, List[float]] - Per-component latencies
        start_times: Dict[str, float] - Active timers
    """
```

**Methods**:

```python
def start_timer(self, name: str) -> None:
    """
    Start a named performance timer.
    
    Args:
        name: Component identifier (e.g., "yolo", "vlm", "stt")
    """
```

```python
def stop_timer(self, name: str) -> float:
    """
    Stop timer and record duration.
    
    Args:
        name: Component identifier matching start_timer call
        
    Returns:
        Duration in milliseconds
        
    Side Effects:
        Logs if duration > 500ms
    """
```

```python
def plot_metrics(self, output_path: str = "plots/performance_summary.png") -> None:
    """
    Generate performance visualization chart.
    
    Args:
        output_path: Where to save the plot
        
    Creates:
        Box plot showing latency distributions for all tracked components
    """
```

---

## Type Definitions

### Common Types

```python
from typing import TypedDict, List, Optional

class Detection(TypedDict):
    """YOLO detection dictionary."""
    label: str
    confidence: float
    bbox: List[List[float]]

class SpatialEvent(TypedDict):
    """Spatial tracking event."""
    type: str  # "NEW_OBJECT" | "OBJECT_MOVED"
    track_id: int
    label: str
    direction: str  # "left" | "center" | "right"
    distance: str  # "very close" | "close" | "medium distance" | "far"
    confidence: float
    timestamp: float
```

---

## Usage Examples

### Complete Pipeline Example

```python
from perception_layer.camera import Camera
from perception_layer.detector import YoloDetector
from perception_layer.rules import SafetyRules
from reasoning_layer.vlm import QwenVLM
from fusion_layer.engine import FusionEngine
from interaction_layer.tts import TTSEngine
from interaction_layer.stt import ListeningLayer

# Initialize components
camera = Camera()
detector = YoloDetector()
safety = SafetyRules()
qwen = QwenVLM()
tts = TTSEngine()
fusion = FusionEngine(tts)
listener = ListeningLayer(tts, fusion)

# Main loop
for frame in camera.stream():
    # Detect objects
    detections = detector.detect(frame)
    
    # Check safety
    result = safety.evaluate(detections)
    if result:
        severity, message = result
        fusion.handle_safety_alert(message, severity)
    
    # Update spatial context
    fusion.update_spatial_context(detections, time.time())
    
    # Get scene description (periodically)
    if should_sample():
        description = qwen.describe_scene(frame, context_str)
        fusion.handle_vlm_description(description)
    
    # Handle user query (if any)
    if key_pressed('L'):
        listener.listen_for_query()
```

---

## Error Handling

All major methods include try-except blocks with appropriate fallbacks:

```python
try:
    result = model.predict(frame)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    return default_value
```

Common exceptions:
- `torch.cuda.OutOfMemoryError`: GPU memory exhausted
- `requests.ConnectionError`: API backend offline
- `sr.UnknownValueError`: Speech not recognized
- `cv2.error`: Invalid frame format

---

## Performance Best Practices

1. **Use GPU when available**:
   ```json
   "detector.device": "cuda",
   "stt.providers.whisper_local.device": "cuda"
   ```

2. **Tune sampling intervals**:
   ```json
   "perception.sampling_interval": 150  # Run VLM every 150 frames
   ```

3. **Enable redundancy filtering**:
   ```json
   "safety.suppression.enabled": true
   ```

4. **Use faster models**:
   - YOLOv8n instead of YOLOv11m
   - Whisper base instead of large
   - Smaller LLM models (Gemma3:270m, Phi-4)

---

## Logging & Debugging

### Log Levels

- `logger.debug()`: Internal state (not shown in terminal)
- `logger.info()`: Key events (USER, AI interactions)
- `logger.warning()`: Safety alerts
- `logger.error()`: Exceptions and failures

### Viewing Logs

```python
# Terminal (filtered)
# Shows only: USER:, AI:, Safety Alert:, System

# File (complete)
tail -f logs/performance.log
```

---

## Configuration Reference

See `config.json` for all available settings:

- `vlm.*`: Vision-language model backend
- `llm.*`: Language model backend
- `stt.*`: Speech-to-text provider
- `tts.*`: Text-to-speech settings
- `detector.*`: YOLO configuration
- `camera.*`: Video capture settings
- `microphone.*`: Audio input settings
- `safety.*`: Alert behavior
- `perception.*`: Sampling and thresholds
