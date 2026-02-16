# WalkSense System Architecture

## System Block Diagram

This visual reference establishes the structural relationship between all components in the WalkSense system.

```mermaid
graph TB
    %% STYLING
    classDef hardware fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef fast fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef slow fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#c62828,stroke-width:2px;

    %% ==========================================
    %% 1. HARDWARE LAYER
    %% ==========================================
    subgraph Hardware ["1. Hardware Layer"]
        CAM[Camera Source]:::hardware
        MIC[Microphone]:::hardware
    end

    %% ==========================================
    %% 2. PERCEPTION LAYER (Real-time Processing)
    %% ==========================================
    subgraph Perception ["2. Perception Layer (30 Hz)"]
        direction TB
        YOLO[YOLO Detector<br/>(Object Detection)]:::fast
        SAFETY[Safety Rules<br/>(Deterministic Checks)]:::fast
        STT[STT Listener<br/>(Speech-to-Text)]:::fast
    end

    %% ==========================================
    %% 3. REASONING LAYER (Asynchronous AI)
    %% ==========================================
    subgraph Reasoning ["3. Reasoning Layer (0.2 Hz)"]
        direction TB
        VLM[Qwen VLM<br/>(Visual Understanding)]:::slow
        LLM[LLM Reasoner<br/>(Complex Logic)]:::slow
    end

    %% ==========================================
    %% 4. FUSION LAYER (The "Brain")
    %% ==========================================
    subgraph Fusion ["4. Fusion Layer (Orchestration)"]
        direction TB
        FUSION[Fusion Engine]:::core
        SPATIAL[Spatial Awareness<br/>(Tracking & Memory)]:::core
        ROUTER[Decision Router<br/>(Priority Handling)]:::core
        CTX[Context Manager<br/>(History & Redundancy)]:::core
    end

    %% ==========================================
    %% 5. INTERACTION LAYER (Output)
    %% ==========================================
    subgraph Interaction ["5. Interaction Layer"]
        TTS[TTS Controller]:::output
        WORKER[Persistent Audio Process<br/>(Subprocess)]:::output
        HAPTIC[Haptics / Buzzer]:::output
        SPEAKER((Speaker)):::output
    end

    %% ==========================================
    %% DATA FLOW CONNECTIONS
    %% ==========================================
    
    %% Video Path
    CAM -->|RGB Frame| YOLO
    YOLO -->|Detections List| SAFETY
    YOLO -->|Detections List| SPATIAL
    
    %% Audio Path
    MIC -->|Audio Stream| STT
    STT -->|User Query Text| FUSION
    
    %% Safety Logic (Fast Path)
    SAFETY -->|Critical Alerts| FUSION
    
    %% AI Logic (Slow Path)
    FUSION -->|Sampled Frame| VLM
    VLM -->|Scene Description| FUSION
    FUSION -->|Query + Scene + Spatial| LLM
    LLM -->|Natural Language Answer| FUSION
    
    %% Decision & Output
    SPATIAL -->|Directional Warnings| FUSION
    FUSION -->|Alert / Response Event| ROUTER
    
    ROUTER -->|Check Redundancy| CTX
    CTX -->|Approved| ROUTER
    
    ROUTER -->|Text| TTS
    ROUTER -->|Signal| HAPTIC
    
    TTS -->|Pipe (stdin)| WORKER
    WORKER -->|Sound Wave| SPEAKER
```

## Layer Responsibilities

| Layer | Responsibility | Key Components | Latency |
|-------|---------------|----------------|---------|
| **Hardware** | Raw Data Acquisition | Camera, Mic | 0ms |
| **Perception** | Immediate Feature Extraction | YOLOv8, Whisper | ~30ms |
| **Fusion** | Context Integration & Decision Making | FusionEngine, DecisionRouter | <5ms |
| **Reasoning** | Deep Understanding (VLM/LLM) | Qwen, Llama/Gemma | 2000-5000ms |
| **Interaction** | User Feedback Generation | TTSWrapper, AudioWorker | ~100ms |

## Critical Data Paths

1.  **The Safety Loop (Fast)**:
    `Camera -> YOLO -> SafetyRules -> Fusion -> Router -> TTS`
    *Priority: HIGHEST. Must complete in <100ms.*

2.  **The Reasoning Loop (Slow)**:
    `Camera -> VLM -> Fusion -> LLM -> Router -> TTS`
    *Priority: LOW. Runs in background. Provides "System Intelligence".*

3.  **The Query Loop (User Initiated)**:
    `Mic -> STT -> Fusion -> (Wait for VLM) -> LLM -> TTS`
    *Priority: HIGH. Triggered on demand.*
