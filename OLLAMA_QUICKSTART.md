# 🚀 Quick Start with Your Ollama Models

## Your Models
- **qwen3-vl:2b** (1.9 GB) - Vision model for scene understanding
- **gemma3:270m** (291 MB) - Fast text model for query answering

## ✅ Start Ollama

```bash
# Make sure Ollama is running
ollama serve
```

## 📝 Configuration (Already Set!)

The system is pre-configured to use your models in `scripts/run_enhanced_camera.py`:

```python
# VISION MODEL - For scene description  
VLM_BACKEND = "ollama"
VLM_MODEL = "qwen3-vl:2b"    # ← Your vision model

# TEXT MODEL - For answering queries
LLM_BACKEND = "ollama"  
LLM_MODEL = "gemma3:270m"    # ← Your text model (super fast!)
```

## 🎯 Run the System

```bash
cd /home/bot/repos/WalkSense

# Activate virtual environment  
source venv/bin/activate  

# Run enhanced demo with your models
python scripts/run_enhanced_camera.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `S` | **Start** system |
| `L` | **Ask question** - Try: "What's around me?" |
| `M` | **Mute/Unmute** audio |
| `Q` | **Quit** |

## 🔧 Easy Model Switching

To try different combinations, edit `scripts/run_enhanced_camera.py` (lines 140-147):

### Option 1: Use Vision Model for Everything
```python
VLM_MODEL = "qwen3-vl:2b"
LLM_MODEL = "qwen3-vl:2b"  # Use same model for queries too
```

### Option 2: Mix Ollama + LM Studio
```python
VLM_BACKEND = "ollama"
VLM_MODEL = "qwen3-vl:2b"

LLM_BACKEND = "lm_studio"          # Different backend!
LLM_URL = "http://localhost:1234/v1"
LLM_MODEL = "llama-3-8b"
```

### Option 3: Different Ollama Models
```bash
# Pull more models first
ollama pull llama3:8b
ollama pull mistral:7b
```

Then configure:
```python
LLM_MODEL = "llama3:8b"  # Better quality queries
# OR
LLM_MODEL = "mistral:7b"  # Fast queries
```

## 📊 Expected Performance
- **Vision (qwen3-vl:2b)**: 2-5 seconds per scene description
- **Text (gemma3:270m)**: ~1 second per query (very fast!)

## ✨ Test It

1. **Start the system**: Press `S`
2. **Hold something up to camera** (phone, book, cup)
3. **Press `L` and ask**: "What's in front of me?"
4. **Listen**: gemma3 will answer based on qwen3-vl's vision!

## 🔍 Troubleshooting

### "Could not connect to Ollama"
```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve
```

### "Model not found"
```bash
# Verify models are installed
ollama list

# Should show:
# qwen3-vl:2b
# gemma3:270m
```

### "Vision model too slow"
- Reduce frame sampling: Edit line 183
  ```python
  sampler = FrameSampler(every_n_frames=200)  # Process less frequently
  ```

## 💡 Pro Tips

1. **gemma3:270m is tiny and FAST** - Perfect for quick queries!
2. **qwen3-vl:2b is optimized for vision** - Best for scene understanding
3. **Keep Ollama running** in the background
4. **Test without camera first**: 
   ```bash
   ollama run qwen3-vl:2b "Describe this image" < test.jpg
   ```

---

**Everything is configured and ready to go! Just run:**
```bash
python scripts/run_enhanced_camera.py
```
