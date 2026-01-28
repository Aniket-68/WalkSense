# WalkSense Optimization - Implementation Plan

## Overview

This plan addresses the major bottlenecks identified in the WalkSense project:
1. **GPU Acceleration** - Reduce latency by 60-70%
2. **LangChain Integration** - Improve architecture and maintainability
3. **VLM Fine-tuning** - Better task-specific performance
4. **YOLO Upgrade** - Marginal accuracy/speed improvements

---

## Phase 1: GPU Acceleration & STT Optimization (HIGH PRIORITY)

**Timeline**: 4-5 days  
**Expected Impact**: 60-70% latency reduction

### Proposed Changes

#### 1. YOLO GPU Enablement

**Files to modify**:
- [config.py](file:///d:/Github/WalkSense/config.py)
- [safety/yolo_detector.py](file:///d:/Github/WalkSense/safety/yolo_detector.py)

**Changes**:

```python
# config.py - Update YOLO configuration
YOLO = {
    "model_path": "models/yolo/yolov8n.pt",
    "confidence_threshold": 0.65,
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Auto-detect
    "half_precision": True  # Enable FP16 for GPU
}
```

```python
# safety/yolo_detector.py - Add GPU support
import torch

class YoloDetector:
    def __init__(self, model_path=None, device=None):
        # Auto-detect GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = YOLO(model_path or "models/yolo/yolov8n.pt")
        
        # Enable half precision on GPU
        if self.device == "cuda":
            self.model.to(self.device)
            print(f"[YOLO] Using GPU with FP16 precision")
```

---

#### 2. Faster-Whisper GPU + Persistent Loading

**Files to modify**:
- [interaction/listening_layer.py](file:///d:/Github/WalkSense/interaction/listening_layer.py)
- [config.json](file:///d:/Github/WalkSense/config.json)

**Changes**:

```python
# interaction/listening_layer.py - Pre-load Whisper model
from faster_whisper import WhisperModel

class ListeningLayer:
    def __init__(self, tts_engine, fusion_engine):
        # Load model ONCE at initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8" if self.device == "cuda" else "int8"
        
        print(f"[STT] Loading faster-whisper on {self.device}...")
        self.whisper_model = WhisperModel(
            "base",  # or "small" for better accuracy
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1  # Optimize for single inference
        )
        print(f"[STT] Model loaded successfully")
    
    def _recognize_faster_whisper(self, audio):
        """Transcribe using pre-loaded model"""
        # Model already in memory - instant inference
        segments, info = self.whisper_model.transcribe(
            audio,
            language="en",  # Or "hi" for Hindi
            beam_size=1,  # Faster inference
            vad_filter=True  # Skip silence
        )
        
        text = " ".join([s.text for s in segments])
        return text, info.language, info.language_probability
```

---

#### 3. VLM Quantization

**Files to modify**:
- [reasoning/qwen_vlm.py](file:///d:/Github/WalkSense/reasoning/qwen_vlm.py)
- [config.py](file:///d:/Github/WalkSense/config.py)

**Changes**:

```python
# config.py - Add quantization config
VISION_MODEL = {
    "backend": "ollama",
    "model_name": "qwen3-vl:2b",
    "ollama_url": "http://localhost:11434",
    "lm_studio_url": "http://localhost:1234/v1",
    
    "huggingface": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "device": "cuda",
        "torch_dtype": "float16",
        "quantization": "int8"  # NEW: Enable quantization
    }
}
```

```python
# reasoning/qwen_vlm.py - Add INT8 quantization
from transformers import BitsAndBytesConfig

def _init_huggingface(self):
    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    
    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        self.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config  # Apply quantization
    )
```

---

### Verification Plan

1. **GPU Availability Check**:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
   ```

2. **Benchmark Script**:
   ```python
   # scripts/benchmark_phase1.py
   import time
   import cv2
   from safety.yolo_detector import YoloDetector
   from interaction.listening_layer import ListeningLayer
   
   # Test YOLO speed
   detector = YoloDetector()
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   
   times = []
   for _ in range(100):
       start = time.time()
       detections = detector.detect(frame)
       times.append((time.time() - start) * 1000)
   
   print(f"YOLO Average: {sum(times)/len(times):.2f}ms")
   cap.release()
   ```

3. **Target Metrics**:
   - YOLO: < 25ms per frame
   - STT initialization: < 5s
   - STT inference: < 2s per query

---

## Phase 2: LangChain/LangGraph Integration (MEDIUM-HIGH PRIORITY)

**Timeline**: 5-7 days  
**Expected Impact**: Better maintainability, easier feature additions

### Proposed Changes

#### 1. Install Dependencies

```bash
pip install langchain langgraph langchain-ollama langchain-openai
```

---

#### 2. Create LangGraph Workflow

**New File**: [orchestration/walksense_graph.py](file:///d:/Github/WalkSense/orchestration/walksense_graph.py)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
import numpy as np

# State definition
class WalkSenseState(TypedDict):
    # Inputs
    frame: np.ndarray
    timestamp: float
    user_query: Optional[str]
    
    # Detection outputs
    detections: List[dict]
    spatial_context: str
    
    # VLM outputs
    scene_description: Optional[str]
    vlm_triggered: bool
    
    # LLM outputs
    llm_response: Optional[str]
    
    # Safety outputs
    safety_alert: Optional[str]
    alert_priority: str
    
    # Output action
    tts_message: Optional[str]

# Node functions
def yolo_detection_node(state: WalkSenseState) -> WalkSenseState:
    """Run YOLO object detection"""
    from safety.yolo_detector import YoloDetector
    
    detector = YoloDetector()
    detections = detector.detect(state["frame"])
    state["detections"] = detections
    return state

def spatial_tracking_node(state: WalkSenseState) -> WalkSenseState:
    """Update spatial context with detections"""
    from inference.spatial_context_manager import SpatialContextManager
    
    spatial_mgr = SpatialContextManager()
    spatial_mgr.update(
        state["detections"],
        state["timestamp"],
        state["frame"].shape[1]
    )
    state["spatial_context"] = spatial_mgr.get_context_for_llm()
    return state

def vlm_node(state: WalkSenseState) -> WalkSenseState:
    """Run VLM scene description"""
    from reasoning.qwen_vlm import QwenVLM
    
    qwen = QwenVLM()
    context_str = ", ".join([d["label"] for d in state["detections"]])
    
    if state.get("user_query"):
        context_str += f". USER QUESTION: {state['user_query']}"
    
    description = qwen.describe_scene(state["frame"], context=context_str)
    state["scene_description"] = description
    state["vlm_triggered"] = True
    return state

def llm_reasoning_node(state: WalkSenseState) -> WalkSenseState:
    """LLM-based query answering"""
    from inference.llm_reasoner import LLMReasoner
    
    llm = LLMReasoner()
    response = llm.answer_query(
        user_query=state["user_query"],
        spatial_context=state["spatial_context"],
        scene_description=state.get("scene_description")
    )
    state["llm_response"] = response
    state["tts_message"] = response
    return state

def safety_check_node(state: WalkSenseState) -> WalkSenseState:
    """Evaluate safety rules"""
    from safety.safety_rules import SafetyRules
    
    safety = SafetyRules()
    result = safety.evaluate(state["detections"])
    
    if result:
        alert_type, message = result
        state["safety_alert"] = message
        state["alert_priority"] = alert_type
        state["tts_message"] = message
    
    return state

# Conditional routing
def should_run_vlm(state: WalkSenseState) -> str:
    """Decide if VLM should run"""
    # Always run VLM if user has query
    if state.get("user_query"):
        return "vlm"
    
    # Otherwise check frame sampling
    # (This would be managed by FrameSampler in production)
    return "skip"

def route_by_priority(state: WalkSenseState) -> str:
    """Route based on priority"""
    if state.get("alert_priority") == "CRITICAL_ALERT":
        return "safety"
    elif state.get("user_query"):
        return "llm"
    else:
        return "output"

# Build workflow
def create_walksense_workflow():
    workflow = StateGraph(WalkSenseState)
    
    # Add nodes
    workflow.add_node("yolo", yolo_detection_node)
    workflow.add_node("spatial", spatial_tracking_node)
    workflow.add_node("vlm", vlm_node)
    workflow.add_node("llm", llm_reasoning_node)
    workflow.add_node("safety", safety_check_node)
    
    # Define flow
    workflow.set_entry_point("yolo")
    workflow.add_edge("yolo", "spatial")
    
    # Conditional VLM trigger
    workflow.add_conditional_edges(
        "spatial",
        should_run_vlm,
        {
            "vlm": "vlm",
            "skip": "safety"
        }
    )
    
    workflow.add_edge("vlm", "safety")
    
    # Priority routing after safety
    workflow.add_conditional_edges(
        "safety",
        route_by_priority,
        {
            "safety": END,
            "llm": "llm",
            "output": END
        }
    )
    
    workflow.add_edge("llm", END)
    
    return workflow.compile()

# Usage in main loop
def process_frame_with_langraph(frame, timestamp, user_query=None):
    workflow = create_walksense_workflow()
    
    initial_state = {
        "frame": frame,
        "timestamp": timestamp,
        "user_query": user_query,
        "detections": [],
        "spatial_context": "",
        "scene_description": None,
        "llm_response": None,
        "safety_alert": None,
        "alert_priority": "NORMAL",
        "tts_message": None,
        "vlm_triggered": False
    }
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state
```

---

#### 3. Integrate into Main Loop

**File to modify**: [scripts/run_enhanced_camera.py](file:///d:/Github/WalkSense/scripts/run_enhanced_camera.py)

Replace main loop with:

```python
from orchestration.walksense_graph import process_frame_with_langraph

for frame in camera.stream():
    current_time = time.time()
    
    # Process frame through LangGraph
    result = process_frame_with_langraph(
        frame=frame,
        timestamp=current_time,
        user_query=current_user_query
    )
    
    # Handle output
    if result.get("tts_message"):
        tts.speak(result["tts_message"])
    
    # Visualize
    frame = draw_detections(frame, result["detections"])
    frame = draw_overlay(frame, result["spatial_context"], result.get("scene_description"))
    
    cv2.imshow("WalkSense", frame)
```

---

### Verification Plan

1. **Test workflow execution**
2. **Verify all nodes execute**
3. **Check priority routing works**
4. **Benchmark latency (should be similar or better)**

---

## Phase 3: VLM Fine-tuning (MEDIUM PRIORITY)

**Timeline**: 1-2 weeks  
**Expected Impact**: 20-30% faster, better task accuracy

### Data Collection

1. **Collect 500-1000 navigation scenes**:
   - Indoor environments
   - Outdoor sidewalks
   - Obstacle scenarios
   
2. **Annotate with**:
   - Object descriptions
   - Safety hazards
   - Navigation hints

### Fine-tuning Setup

**New File**: [training/finetune_vlm.py](file:///d:/Github/WalkSense/training/finetune_vlm.py)

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, Trainer
from datasets import Dataset

# Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Prepare dataset
dataset = Dataset.from_dict({
    "image": image_paths,
    "text": annotations
})

# Training config
training_args = {
    "output_dir": "./models/qwen_finetuned",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-5,
    "fp16": True
}

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("./models/qwen_navigation")
```

---

## Phase 4: YOLO v11 Upgrade (LOW-MEDIUM PRIORITY)

**Timeline**: 2-3 days  
**Expected Impact**: 15-20% faster + better accuracy

### Changes

**File to modify**: [config.py](file:///d:/Github/WalkSense/config.py)

```python
YOLO = {
    "model_path": "models/yolo/yolo11n.pt",  # Upgrade to v11
    "confidence_threshold": 0.65,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "half_precision": True
}
```

**Download model**:
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); model.save('models/yolo/yolo11n.pt')"
```

### Benchmark

Compare v8n vs v11n on test footage:
- Speed
- Accuracy (mAP)
- False positives

---

## Rollback Procedures

For each phase, maintain backup:

```bash
# Before Phase 1
git checkout -b backup-pre-phase1
git add .
git commit -m "Backup before GPU optimization"

# If rollback needed
git checkout backup-pre-phase1
```

---

## Success Metrics

| Phase | Metric | Current | Target |
|-------|--------|---------|--------|
| Phase 1 | YOLO Latency | 85ms | 20ms |
| Phase 1 | STT Latency | 10s | 2s |
| Phase 1 | VLM Latency | 3-5s | 1-1.5s |
| Phase 2 | Code Complexity | High | Low |
| Phase 2 | Feature Addition Time | Days | Hours |
| Phase 3 | VLM Relevance | 70% | 90% |
| Phase 4 | YOLO mAP | 37.3 | 39.5 |

---

## Risks & Mitigations

1. **GPU unavailable on deployment**
   - Mitigation: Auto-fallback to CPU
   
2. **Fine-tuned VLM underperforms**
   - Mitigation: Keep original model as fallback
   
3. **LangChain adds overhead**
   - Mitigation: Benchmark each phase, revert if slower

---

## Dependencies

```bash
# Phase 1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes

# Phase 2  
pip install langchain langgraph langchain-ollama

# Phase 3
pip install datasets accelerate peft

# Phase 4
pip install ultralytics>=8.2.0
```
