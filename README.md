# WalkSense

WalkSense is an AI-powered assistive system for the visually impaired, combining real-time object detection, spatial tracking, and LLM reasoning to provide context-aware safety alerts and scene understanding.

## Features
- **Real-time Object Detection**: Uses YOLO (v8n/11m) to identify hazards and obstacles.
- **Spatial Awareness**: Tracks object movement, distance, and direction.
- **Semantic Scene Understanding**: Uses Qwen-VL to describe complex visual environments.
- **Natural Language Interaction**: AI handles user queries about the surroundings using an LLM (Qwen/Llama).
- **Voice Feedback**: Real-time TTS alerts and reasoning responses.

## Quick Start

### 1. Installation
The easiest way to set up WalkSense is using the provided setup script:

```bash
# Clone the repository
git clone https://github.com/Aniket-68/WalkSense.git
cd WalkSense

# Run the comprehensive setup script
python scripts/setup_project.py
```

This script will:
- Create a virtual environment (`venv`).
- Install all dependencies from `requirements.txt`.
- Download necessary models (YOLOv8n, YOLO11m, Whisper base).

### 2. Configuration
Edit `config.json` to customize backends (LLM, VLM) and hardware settings.

### 3. Run the System
To start the enhanced WalkSense system:

```bash
# Activate the environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the demo
python scripts/run_enhanced_camera.py
```

## Hardware Requirements
- Camera (USB or integrated)
- Microphone (for voice queries)
- Speakers/Headphones (for audio feedback)

## Controls
- `S`: Start system
- `L`: Ask question (Push-to-talk)
- `M`: Toggle mute
- `Q`: Quit
