# User Query Flow: VLM + LLM Pipeline

## Quick Answer

**User queries use BOTH VLM and LLM in a two-stage pipeline:**

1. **VLM** (Vision-Language Model) - Describes what the camera sees
2. **LLM** (Language Model) - Reasons over the VLM description to answer the query

## Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER ASKS A QUESTION                         │
│              (e.g., "What obstacles are ahead?")                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: handle_user_query()                                    │
│  ─────────────────────────────────────────────────────────────  │
│  • Sets self.pending_query = "What obstacles are ahead?"        │
│  • Sends acknowledgment: "Checking on: What obstacles..."       │
│  • TTS speaks the acknowledgment                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ (Query is now PENDING)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Wait for next VLM description                          │
│  ─────────────────────────────────────────────────────────────  │
│  • System continues normal operation                            │
│  • Camera captures frames                                       │
│  • YOLO detects objects                                         │
│  • When sampling interval triggers...                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: VLM Processes Frame                                    │
│  ─────────────────────────────────────────────────────────────  │
│  • QwenVLM.describe_scene(frame, context)                       │
│  • Context includes: detected objects + USER QUESTION           │
│  • VLM generates visual description focused on the query        │
│  • Example: "Scene shows a chair 2 meters ahead, table on       │
│    left, and clear path on right"                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: handle_vlm_description(vlm_text)                       │
│  ─────────────────────────────────────────────────────────────  │
│  • Checks: if self.pending_query exists?                        │
│  • YES → Go to Step 5 (LLM reasoning)                           │
│  • NO  → Just speak the scene description                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ (pending_query exists!)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: _generate_llm_answer(query, vlm_desc)                  │
│  ─────────────────────────────────────────────────────────────  │
│  • Gathers context:                                             │
│    - User query: "What obstacles are ahead?"                    │
│    - VLM description: "chair 2m ahead, table left..."           │
│    - Spatial context: object tracking, recent events            │
│                                                                  │
│  • Calls LLM.answer_query() with all context                    │
│                                                                  │
│  • LLM reasons and generates answer:                            │
│    "There is a chair directly ahead at 2 meters. A table is     │
│     on your left. The right side has a clear path."             │
│                                                                  │
│  • Performs hallucination check (numeric grounding)             │
│  • Routes answer to TTS                                         │
│  • Clears pending_query                                         │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: User Hears Answer                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • TTS speaks the LLM-generated answer                          │
│  • Answer is grounded in visual evidence from VLM               │
│  • Answer is contextualized by LLM reasoning                    │
└─────────────────────────────────────────────────────────────────┘
```

## Code References

### 1. User Query Received
**File:** `scripts/run_enhanced_camera.py` (line 369)
```python
fusion.handle_user_query(query)
```

**File:** `fusion_layer/engine.py` (lines 140-157)
```python
def handle_user_query(self, query: str):
    self.pending_query = query  # Store query
    ack_event = AlertEvent("RESPONSE", f"Checking on: {query}")
    self.router.route(ack_event)  # Speak acknowledgment
```

### 2. VLM Triggered with Query Context
**File:** `scripts/run_enhanced_camera.py` (lines 447-449)
```python
context_str = ", ".join([d["label"] for d in detections])
if has_query:
    context_str += f". USER QUESTION: {current_user_query}"
```

**File:** `reasoning_layer/vlm.py`
```python
def describe_scene(self, frame, context=""):
    # VLM receives: "person, chair, table. USER QUESTION: What obstacles are ahead?"
    # VLM generates description focused on answering the question
```

### 3. VLM Description Processed
**File:** `fusion_layer/engine.py` (lines 100-130)
```python
def handle_vlm_description(self, text: str):
    self.spatial.add_scene_description(text)
    
    if self.pending_query:
        # Query exists! Use LLM to answer it
        ans = self._generate_llm_answer(self.pending_query, text)
        self.pending_query = None
        return ans
    else:
        # No query, just speak the scene description
        event = AlertEvent("SCENE_DESC", text)
        self.router.route(event)
        return text
```

### 4. LLM Generates Answer
**File:** `fusion_layer/engine.py` (lines 159-208)
```python
def _generate_llm_answer(self, query: str, vlm_desc: str):
    spatial_ctx = self.spatial.get_context_for_llm()
    
    # LLM reasons over:
    # - User query
    # - VLM visual description
    # - Spatial tracking data
    answer = self.llm.answer_query(
        user_query=query,
        spatial_context=spatial_ctx,
        scene_description=vlm_desc
    )
    
    # Anti-hallucination check
    # ... (numeric grounding)
    
    # Route answer to TTS
    self.router.route(AlertEvent("RESPONSE", answer))
    return answer
```

**File:** `reasoning_layer/llm.py`
```python
def answer_query(self, user_query, spatial_context, scene_description):
    # LLM prompt includes all context
    # Returns natural language answer grounded in visual evidence
```

## Why This Two-Stage Approach?

### VLM (Vision) - Stage 1
- **Strength**: Understands visual content
- **Role**: "What does the camera see?"
- **Output**: Visual description grounded in the image
- **Example**: "Scene shows a chair 2 meters ahead, table on left"

### LLM (Reasoning) - Stage 2
- **Strength**: Natural language reasoning
- **Role**: "How does this answer the user's question?"
- **Output**: Contextualized answer in natural language
- **Example**: "There is a chair directly ahead. I recommend going right."

### Benefits of VLM → LLM Pipeline

1. **Visual Grounding**: VLM ensures answers are based on what's actually visible
2. **Natural Responses**: LLM formats answers in conversational language
3. **Contextual Reasoning**: LLM can combine visual info with spatial tracking
4. **Hallucination Prevention**: VLM provides factual visual baseline
5. **Flexibility**: LLM can answer complex questions using simple VLM descriptions

## Example Walkthrough

**User asks:** "How many people are in front of me?"

1. **Query stored**: `pending_query = "How many people are in front of me?"`
2. **Acknowledgment**: TTS speaks "Checking on: How many people are in front of me?"
3. **VLM triggered**: Next frame is sent to VLM with context:
   - Detected objects: "person, person, chair"
   - User question: "How many people are in front of me?"
4. **VLM responds**: "Two people standing in the center of the frame, approximately 3 meters away"
5. **LLM processes**:
   - Query: "How many people are in front of me?"
   - VLM: "Two people standing in the center..."
   - Spatial: "person center 3m, person center 3m"
6. **LLM answers**: "There are 2 people directly in front of you, about 3 meters away"
7. **TTS speaks**: User hears the answer

## Configuration

### VLM Settings (config.json)
```json
"vlm": {
    "active_provider": "lm_studio",
    "providers": {
        "lm_studio": {
            "url": "http://localhost:1234/v1",
            "model_id": "qwen/qwen3-vl-4b"
        }
    }
}
```

### LLM Settings (config.json)
```json
"llm": {
    "active_provider": "ollama",
    "providers": {
        "ollama": {
            "url": "http://localhost:11434",
            "model_id": "gemma3:270m"
        }
    }
}
```

## Performance Notes

- **VLM latency**: ~2-5 seconds (depends on model size and GPU)
- **LLM latency**: ~0.5-2 seconds (faster, text-only reasoning)
- **Total response time**: ~3-7 seconds from question to answer

The system is designed to be **visually grounded** - the LLM never makes up information about what's in the scene. It only reasons over what the VLM actually sees.
