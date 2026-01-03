# config.py
"""
WalkSense Configuration File

Edit this file to change models, backends, and system parameters.
All settings are in one place for easy management.
"""

#==============================================================================
# 🎯 MODEL CONFIGURATION - EDIT HERE TO CHANGE MODELS
#==============================================================================

# VISION MODEL (VLM) - For scene understanding
VISION_MODEL = {
    "backend": "ollama",                    # Options: "ollama", "lm_studio", "huggingface"
    "model_name": "qwen3-vl:2b",           # Your vision model
    "ollama_url": "http://localhost:11434",
    "lm_studio_url": "http://localhost:1234/v1",
    
    # HuggingFace settings (only if backend is "huggingface")
    "huggingface": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "device": "cuda",                   # "cuda" or "cpu"
        "torch_dtype": "float16"            # "float16" or "float32"
    }
}

# TEXT MODEL (LLM) - For query answering
TEXT_MODEL = {
    "backend": "ollama",                    # Options: "ollama", "lm_studio"
    "model_name": "gemma3:270m",           # Your text model (fast!)
    "ollama_url": "http://localhost:11434",
    "lm_studio_url": "http://localhost:1234/v1",
    "temperature": 0.7,                     # 0.0 = deterministic, 1.0 = creative
    "max_tokens": 100                       # Maximum response length
}

#==============================================================================
# 🔧 QUICK PRESETS - Uncomment to use
#==============================================================================

# # PRESET 1: All Ollama (Current Setup)
# VISION_MODEL["backend"] = "ollama"
# VISION_MODEL["model_name"] = "qwen3-vl:2b"
# TEXT_MODEL["backend"] = "ollama"
# TEXT_MODEL["model_name"] = "gemma3:270m"

# # PRESET 2: All LM Studio
# VISION_MODEL["backend"] = "lm_studio"
# VISION_MODEL["model_name"] = "local-model"
# TEXT_MODEL["backend"] = "lm_studio"
# TEXT_MODEL["model_name"] = "local-model"

# # PRESET 3: Mixed - Ollama Vision + LM Studio Text
# VISION_MODEL["backend"] = "ollama"
# VISION_MODEL["model_name"] = "qwen3-vl:2b"
# TEXT_MODEL["backend"] = "lm_studio"
# TEXT_MODEL["model_name"] = "llama-3-8b"

#==============================================================================
# 📷 CAMERA & FRAME PROCESSING - Navigation mode
#==============================================================================

CAMERA = {
    "id": 0,                                # Camera device ID (0 = default)
    "width": 1280,
    "height": 720
}

FRAME_PROCESSING = {
    "sample_every_n_frames": 600,          # VLM every 600 frames (~20 seconds at 30fps)
    "scene_change_threshold": 0.35         # Even less sensitive - only major scene changes
}

#==============================================================================
# 🎯 OBJECT DETECTION (YOLO)
#==============================================================================

YOLO = {
    "model_path": "models/yolo/yolov8n.pt",
    "confidence_threshold": 0.65,           # Higher confidence = fewer false positives
    "device": "cpu"                         # "cuda" or "cpu"
}

#==============================================================================
# 📍 SPATIAL TRACKING - Human-paced navigation
#==============================================================================

SPATIAL_TRACKING = {
    "movement_threshold": 150.0,            # Much higher - only very significant movements
    "time_threshold": 30.0,                 # 30 seconds minimum between announcements
    "max_history": 5,                       # Remember fewer events
    
    # IoU Tracker
    "iou_threshold": 0.3,                   # Minimum IoU for track matching
    "max_age": 30                           # Frames - drop track if not seen
}

#==============================================================================
# 🔊 AUDIO
#==============================================================================

AUDIO = {
    "tts_enabled": True,
    "tts_rate": 150,                        # Words per minute
    "stt_enabled": False,                   # Speech-to-text
    "stt_language": "en-US",
    "stt_timeout": 5                        # Seconds to listen
}

#==============================================================================
# 🚨 SAFETY RULES
#==============================================================================

SAFETY = {
    "alert_cooldown": 40.0,                 # 40 seconds between same alerts (very patient)
    
    "critical_objects": [
        "fire", "stairs"                    # Only real immediate dangers
    ],
    
    "warning_objects": [
        "car", "truck", "bus",              # Large vehicles only
        "person"                            # People
    ]
}

#==============================================================================
# 🎙️ LLM NARRATION - Intelligent guidance
#==============================================================================

LLM_NARRATION = {
    "enabled": True,                        # Use LLM to narrate environment
    "interval": 2.0,                        # Seconds between navigation updates
    "max_length": 15,                       # Maximum words per announcement
    "mode": "navigation"                    # Style: "navigation", "descriptive", "minimal"
}

#==============================================================================
# ⚙️ SYSTEM
#==============================================================================

SYSTEM = {
    "start_muted": False,                   # Start with audio muted
    "show_visualization": True,             # Show camera window
    "log_level": "INFO",                    # DEBUG, INFO, WARNING, ERROR
    "async_vlm": True,                      # Run VLM in separate thread
    "vlm_timeout": 30                       # Seconds - VLM inference timeout
}

#==============================================================================
# 🎨 VISUALIZATION (UI)
#==============================================================================

VISUALIZATION = {
    "window_name": "WalkSense Enhanced",
    "window_width": 1280,
    "window_height": 720,
    
    # Colors (BGR format)
    "colors": {
        "critical": (0, 0, 255),            # Red
        "warning": (0, 255, 255),           # Yellow
        "info": (255, 0, 0),                # Blue
        "safe": (0, 255, 0)                 # Green
    }
}

#==============================================================================
# 🔥 QUICK MODEL EXAMPLES
#==============================================================================

# Available Ollama Models (your system):
#   - qwe n3-vl:2b   (1.9 GB) - Vision model
#   - gemma3:270m   (291 MB) - Super fast text model
#
# To add more Ollama models:
#   ollama pull llama3:8b
#   ollama pull mistral:7b
#   ollama pull llava:7b
#
# Then update above:
#   TEXT_MODEL["model_name"] = "llama3:8b"

#==============================================================================
# 📋 HELPER FUNCTIONS
#==============================================================================

def get_config_summary():
    """Print current configuration"""
    print("\n" + "="*70)
    print("WALKSENSE CONFIGURATION")
    print("="*70)
    print(f"\n🖼️  VISION: {VISION_MODEL['backend']} / {VISION_MODEL['model_name']}")
    print(f"💬 TEXT:   {TEXT_MODEL['backend']} / {TEXT_MODEL['model_name']}")
    print(f"🎯 YOLO:   {YOLO['model_path']}")
    print(f"📷 CAMERA: {CAMERA['width']}x{CAMERA['height']}")
    print(f"📍 SPATIAL: {SPATIAL_TRACKING['movement_threshold']}px / {SPATIAL_TRACKING['time_threshold']}s")
    print("="*70 + "\n")


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check backends
    valid_backends = ["ollama", "lm_studio", "huggingface"]
    if VISION_MODEL["backend"] not in valid_backends:
        errors.append(f"Invalid vision backend: {VISION_MODEL['backend']}")
    if TEXT_MODEL["backend"] not in ["ollama", "lm_studio"]:
        errors.append(f"Invalid text backend: {TEXT_MODEL['backend']}")
    
    # Check thresholds
    if not 0 <= FRAME_PROCESSING["scene_change_threshold"] <= 1:
        errors.append("scene_change_threshold must be between 0 and 1")
    
    if errors:
        print("\n⚠️  Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration valid")
    return True


# Auto-validate on import
if __name__ != "__main__":
    validate_config()
