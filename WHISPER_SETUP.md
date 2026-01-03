# OpenAI Whisper STT Integration

## ✅ What's Added

WalkSense now supports **3 STT providers**:

1. **Google** (default, free, requires internet)
2. **OpenAI Whisper API** (cloud, most accurate, requires API key)
3. **Local Whisper** (offline, private, runs on your device)

## 🚀 Setup

### Option 1: OpenAI Whisper API (Recommended for accuracy)

```bash
# Install dependencies
pip install openai>=1.0.0

# Set API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Or add to ~/.bashrc:
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.bashrc
```

Edit `config.json`:
```json
"stt": {
    "active_provider": "whisper_api"
}
```

### Option 2: Local Whisper (Offline, Private)

```bash
# Install faster-whisper (recommended - 4x faster)
pip install faster-whisper>=0.10.0

# OR install OpenAI's original whisper
pip install openai-whisper
```

Edit `config.json`:
```json
"stt": {
    "active_provider": "whisper_local",
    "providers": {
        "whisper_local": {
            "model_size": "base",    // tiny, base, small, medium, large
            "device": "cpu",          // or "cuda" for GPU
            "language": "en"
        }
    }
}
```

### Option 3: Keep Google (Default)

No changes needed - works out of the box.

## 📊 Model Comparison

| Provider | Speed | Accuracy | Cost | Offline | Setup |
|----------|-------|----------|------|---------|-------|
| **Google** | Fast | Good | Free | ❌ | Easy |
| **Whisper API** | Fast | Best | ~$0.006/min | ❌ | API Key |
| **Whisper Local (faster-whisper)** | Medium | Excellent | Free | ✅ | Download model |
| **Whisper Local (openai-whisper)** | Slow | Excellent | Free | ✅ | Download model |

## 🎯 Model Sizes (Local Whisper)

| Size | Parameters | VRAM | Speed | Accuracy |
|------|------------|------|-------|----------|
| `tiny` | 39M | ~1GB | Fastest | 70% |
| `base` | 74M | ~1GB | Fast | 80% |
| `small` | 244M | ~2GB | Medium | 90% |
| `medium` | 769M | ~5GB | Slow | 95% |
| `large` | 1550M | ~10GB | Slowest | 98% |

**Recommended:** `base` for CPU, `small` for GPU

## 🧪 Testing

```bash
# Test your STT setup
python3 -c "
from interaction.listening_layer import STTListener
listener = STTListener()
print('Say something...')
text = listener.listen_once(timeout=5)
print(f'You said: {text}')
"
```

## 🔧 Configuration Examples

### Fast & Accurate (API)
```json
{
    "stt": {
        "active_provider": "whisper_api",
        "providers": {
            "whisper_api": {
                "model": "whisper-1",
                "language": "en",
                "timeout": 10
            }
        }
    }
}
```

### Offline & Private (Local)
```json
{
    "stt": {
        "active_provider": "whisper_local",
        "providers": {
            "whisper_local": {
                "model_size": "base",
                "device": "cpu",
                "language": "en"
            }
        }
    }
}
```

### Multilingual Support
```json
{
    "stt": {
        "active_provider": "whisper_api",
        "providers": {
            "whisper_api": {
                "language": "auto"  // Auto-detect language
            }
        }
    }
}
```

## 💡 Tips

1. **For real-time use:** Use `faster-whisper` with `tiny` or `base` model
2. **For accuracy:** Use Whisper API or local `small` model
3. **For privacy:** Use local models (no internet needed)
4. **For multilingual:** Whisper supports 99 languages!

## 🐛 Troubleshooting

**Problem:** `ModuleNotFoundError: faster_whisper`
```bash
pip install faster-whisper
```

**Problem:** `OPENAI_API_KEY not set`
```bash
export OPENAI_API_KEY="your-key-here"
```

**Problem:** Local model too slow
- Use `tiny` or `base` model
- Or switch to `faster-whisper` (4x faster)

**Problem:** CUDA out of memory
```json
"device": "cpu"  // Use CPU instead
```

## 📝 Usage in Code

The STT system automatically uses the configured provider. No code changes needed!

```python
# Press 'L' key in run_enhanced_camera.py
# System will use whatever provider is configured in config.json
```

## ✅ Done!

Your WalkSense now has state-of-the-art speech recognition! 🎉
