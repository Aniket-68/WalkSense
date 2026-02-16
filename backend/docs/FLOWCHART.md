# WalkSense System Flow & Initialization

## System Initialization & Data Flow Diagram

This diagram illustrates how the WalkSense system boots up, initializes its AI models, and processes data in real-time.

```mermaid
graph TD
    %% ==========================================
    %% 1. INITIALIZATION PHASE
    %% ==========================================
    subgraph Init["1. Initialization Phase (Main Thread)"]
        Start((Start)) --> LoadConfig[Load config.json]
        LoadConfig --> InitHW[Init Hardware<br/>(Camera & Mic)]
        
        subgraph Models["Model Loading"]
            InitHW --> InitYOLO[Load YOLOv8<br/>(Perception)]
            InitYOLO --> InitTTS[Init TTS Engine<br/>(Subprocess)]
            InitTTS --> InitSTT[Init STT Listener<br/>(Lazy Load Model)]
            InitSTT --> InitVLM[Init Qwen VLM<br/>(Connection Check)]
            InitVLM --> InitLLM[Init LLM Reasoner<br/>(Connection Check)]
        end
        
        InitLLM --> InitFusion[Init Fusion Engine]
        InitFusion --> BootWorker[Start Async VLM Worker]
        BootWorker --> SystemReady((System Ready))
    end

    %% ==========================================
    %% 2. RUNTIME LOOPS
    %% ==========================================
    
    SystemReady --> MainLoop
    SystemReady --> AudioLoop
    SystemReady --> VLMWorker

    %% --- MAIN VIDEO LOOP ---
    subgraph Loop["2. Main Video Loop (Sync)"]
        direction TB
        MainLoop(Get Frame) --> YoloInfer[YOLO Reference]
        YoloInfer --> SafetyCheck{Safety Rules}
        
        SafetyCheck -->|Critical/Warning| AlertEvent[Create Alert Event]
        AlertEvent --> FusionIn
        
        SafetyCheck -->|Safe| SpatialUpdate[Update Spatial Context]
        SpatialUpdate --> DrawUI[Draw UI Overlay]
        
        DrawUI --> ShowFrame[Display Frame]
        ShowFrame --> MainLoop
    end

    %% --- AUDIO LISTENING THREAD ---
    subgraph Audio["3. Audio Input (Threaded)"]
        AudioLoop(Listen) -->|Trigger Key/Wake| RecordAudio
        RecordAudio --> STTInfer[STT Inference<br/>(Whisper)]
        STTInfer -->|User Query| QueryEvent[Create Query Event]
        QueryEvent --> FusionIn
    end

    %% --- ASYNC VLM WORKER ---
    subgraph Worker["4. VLM Reasoning (Async Worker)"]
        VLMWorker(Wait for Task) -->|Input Frame| qwenInfer[Qwen VLM Inference]
        qwenInfer -->|Scene Description| DescEvent[Scene Event]
        DescEvent --> FusionIn
    end

    %% ==========================================
    %% 5. FUSION & DECISION
    %% ==========================================
    subgraph Fusion["5. Fusion & Interaction (The Brain)"]
        FusionIn{Event Router}
        
        FusionIn -->|Safety Alert| MuteCheck{Muted?}
        MuteCheck -->|No| TTSSpeak[TTS Speak]
        
        FusionIn -->|User Query| PendingState[Set Pending Query]
        PendingState -->|Wait for VLM| Combine[Fusion: Query + VLM Desc]
        Combine --> LLMInfer[LLM Inference]
        LLMInfer -->|Answer| TTSSpeak
        
        FusionIn -->|Scene Desc| Redundancy{Is Redundant?}
        Redundancy -->|No| TTSSpeak
    end
    
    %% Output
    TTSSpeak --> Speaker((Audio Output))
```

## Detailed Component Interaction

### 1. Initialization
- **Config**: The system starts by reading `config.json` to determine which providers (Local vs Cloud) to use.
- **Lazy vs Eager**: 
  - **YOLO** is loaded immediately as it's critical for safety.
  - **TTS** spawns a persistent subprocess to stay ready.
  - **STT** loads the Whisper model into memory but waits for a trigger to run inference.
  - **VLM/LLM** are typically API clients (e.g., to LM Studio or Ollama), so initialization is just a connection check.

### 2. The Main Loop (Perception)
- Runs at ~30 FPS on the main thread.
- **YOLO** detects objects every frame.
- **Safety Rules** checks these objects immediately. If a "Knife" or "Train" is seen, it bypasses complex reasoning and sends an immediate **CRITICAL ALERT** to the Fusion layer.

### 3. Asynchronous Reasoning
- **VLM (Vision)** is heavy. It runs in a separate thread (`QwenWorker`).
- The Main Loop sends a frame to the worker only when necessary (e.g., scene changed or user asked a question).
- This prevents the UI from freezing while the AI "thinks".

### 4. Fusion Logic
The **Fusion Engine** is the central hub. It receives signals from all inputs:
- **Safety**: "Stop! Car ahead." (High Priority)
- **VLM**: "I see a park with benches." (Low Priority, Passive)
- **User**: "Where is the bench?" (Query)
- **Fusion's Job**: If the user asks "Where is the bench?", Fusion waits for the VLM to confirm the bench's location, then uses the LLM to generate a natural answer like "The bench is 2 meters to your right."
