# WalkSense - Critical Fix Implementation Plan

## 🚨 PRIORITY: Fix CUDA/cuDNN Before Anything Else

The system currently shows:
```
Could not locate cudnn_ops64_9.dll
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

**This blocks ALL GPU acceleration**. We must fix this before any other optimizations.

---

## Phase 1: CUDA Environment Setup (CRITICAL)

**Timeline**: 1-2 hours  
**Impact**: Enables 3-4x overall speedup

### Step 1: Verify Current CUDA Installation

Run these commands:

```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version (PyTorch): {torch.version.cuda}')"
```

**Expected Current Output**:
```
PyTorch Version: 2.x.x
CUDA Available: False  # or True with warnings
CUDA Version (PyTorch): 12.4
```

Take note of the **PyTorch CUDA version**. This determines which cuDNN you need.

---

### Step 2: Download and Install cuDNN

#### Option A: NVIDIA Developer Portal (Official)

1. **Create NVIDIA Developer Account**: https://developer.nvidia.com/login
2. **Download cuDNN**: https://developer.nvidia.com/cudnn-downloads
  
   Select:
   - **cuDNN version**: 9.x for CUDA 12.x
   - **OS**: Windows
   - **CUDA**: Match your PyTorch CUDA version (likely 12.4)

3. **Install Steps**:
   ```bash
   # Extract downloaded ZIP to:
   # C:\Program Files\NVIDIA\CUDNN\v9.x\
   
   # Folder structure should be:
   # C:\Program Files\NVIDIA\CUDNN\v9.x\
   #     ├── bin\       (contains cudnn_ops64_9.dll)
   #     ├── include\
   #     └── lib\
   ```

#### Option B: Conda (Easier)

```bash
conda install -c conda-forge cudnn
```

---

### Step 3: Update System PATH

**Windows**:

1. Open **Environment Variables**:
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"

2. **Edit PATH** (System variables):
   - Add: `C:\Program Files\NVIDIA\CUDNN\v9.x\bin`
   - Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`

3. **Restart Terminal** (or reboot if issues persist)

---

### Step 4: Reinstall PyTorch with Correct CUDA

```bash
# Activate your venv
cd d:\Github\WalkSense
venv\Scripts\activate

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

### Step 5: Verify CUDA + cuDNN Working

Create verification script:

**File**: `d:\Github\WalkSense\scripts\verify_cuda.py`

```python
import torch
import sys

print("="*60)
print("CUDA VERIFICATION")
print("="*60)

# Check PyTorch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # Test simple operation
    try:
        x = torch.rand(5, 3).cuda()
        print(f"\n✅ GPU Tensor Test: PASSED")
        print(f"   Tensor Device: {x.device}")
    except Exception as e:
        print(f"\n❌ GPU Tensor Test: FAILED")
        print(f"   Error: {e}")
else:
    print("\n❌ CUDA NOT AVAILABLE")
    print("   Check NVIDIA drivers and PyTorch installation")
    sys.exit(1)

print("\n" + "="*60)
print("✅ CUDA VERIFICATION COMPLETE")
print("="*60)
```

Run it:
```bash
python scripts/verify_cuda.py
```

**Expected Output** (after fix):
```
============================================================
CUDA VERIFICATION
============================================================

PyTorch Version: 2.x.x+cu124
CUDA Available: True
CUDA Version: 12.4
cuDNN Enabled: True
cuDNN Version: 90100
GPU Device: NVIDIA GeForce RTX 4060
GPU Count: 1

✅ GPU Tensor Test: PASSED
   Tensor Device: cuda:0

============================================================
✅ CUDA VERIFICATION COMPLETE
============================================================
```

---

## Phase 2: Optimize Model Configuration

**Timeline**: 30 minutes  
**Prerequisites**: Phase 1 complete

### Change 1: Switch to YOLO11m

**File**: `d:\Github\WalkSense\config.json`

```json
{
  "detector": {
    "active_model": "yolo11m",  // Change from "yolov8n"
    "device": "cuda",
    "confidence_threshold": 0.25,
    "models": {
      "yolov8n": "models/yolo/yolov8n.pt",
      "yolo11m": "models/yolo/yolo11m.pt"
    }
  }
}
```

**Expected Improvement**:
- Better accuracy (37.3 → 49.0 mAP)
- GPU: ~30ms per frame (vs 85ms on CPU)

---

### Change 2: Enable Whisper GPU

**File**: `d:\Github\WalkSense\config.json`

```json
{
  "stt": {
    "active_provider": "whisper_local",
    "providers": {
      "whisper_local": {
        "model": "base",
        "device": "cuda",           // Add this
        "compute_type": "int8",     // Add this
        "language": "en"
      }
    }
  }
}
```

**Expected Improvement**:
- STT: 10s → 2s (5x faster)

---

### Change 3: Verify VLM GPU Usage

**File**: `d:\Github\WalkSense\config.json`

VLM should already be configured for GPU:

```json
{
  "vlm": {
    "active_provider": "lm_studio",  // or "huggingface"
    "providers": {
      "huggingface": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "device": "cuda",              // Verify this
        "torch_dtype": "float16"
      }
    }
  }
}
```

---

## Phase 3: Test End-to-End

### Benchmark Script

**File**: `d:\Github\WalkSense\scripts\benchmark_after_cuda.py`

```python
import time
import cv2
from perception_layer.detector import YoloDetector
from interaction_layer.stt import STTEngine
from reasoning_layer.vlm import QwenVLM

print("="*60)
print("PERFORMANCE BENCHMARK")
print("="*60)

# Test 1: YOLO
print("\n[1/3] Testing YOLO Detection...")
detector = YoloDetector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

yolo_times = []
for i in range(50):
    start = time.time()
    detections = detector.detect(frame)
    yolo_times.append((time.time() - start) * 1000)
    if i % 10 == 0:
        print(f"  Progress: {i+1}/50")

cap.release()
print(f"  Average: {sum(yolo_times)/len(yolo_times):.2f}ms")
print(f"  Min: {min(yolo_times):.2f}ms | Max: {max(yolo_times):.2f}ms")

# Test 2: STT (if available)
try:
    print("\n[2/3] Testing STT...")
    from interaction_layer.stt import STTEngine
    stt = STTEngine()
    print(f"  ✅ STT Loaded (GPU enabled: {stt.device == 'cuda'})")
except Exception as e:
    print(f"  ⚠️ STT not available: {e}")

# Test 3: VLM
print("\n[3/3] Testing VLM...")
try:
    qwen = QwenVLM()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    start = time.time()
    desc = qwen.describe_scene(frame, context="test")
    vlm_time = (time.time() - start)
    
    print(f"  Time: {vlm_time:.2f}s")
    print(f"  Description: {desc[:100]}...")
except Exception as e:
    print(f"  ⚠️ VLM error: {e}")

print("\n" + "="*60)
print("BENCHMARK COMPLETE")
print("="*60)
```

Run:
```bash
python scripts/benchmark_after_cuda.py
```

**Expected Results** (After CUDA Fix):
```
============================================================
PERFORMANCE BENCHMARK
============================================================

[1/3] Testing YOLO Detection...
  Progress: 1/50
  Progress: 11/50
  Progress: 21/50
  Progress: 31/50
  Progress: 41/50
  Average: 28.45ms
  Min: 25.12ms | Max: 35.89ms

[2/3] Testing STT...
  ✅ STT Loaded (GPU enabled: True)

[3/3] Testing VLM...
  Time: 1.35s
  Description: A person sits at a desk with a laptop. A monitor displays code...

============================================================
BENCHMARK COMPLETE
============================================================
```

---

## Phase 4: Update Documentation

**Files to Update**:

1. **`d:\Github\WalkSense\docs\WALKSENSE_ANALYSIS.md`**
   - Replace old structure references
   - Add new layered architecture
   - Update with CUDA setup section

2. **`d:\Github\WalkSense\docs\IMPLEMENTATION_PLAN.md`**
   - Add Phase 0: CUDA Setup
   - Update file paths to new structure

3. **Create**: `d:\Github\WalkSense\docs\CUDA_SETUP.md`

```markdown
# CUDA Setup Guide for WalkSense

## Prerequisites
- NVIDIA GPU (GTX 1060 or better)
- NVIDIA Drivers (latest)
- Windows 10/11

## Installation Steps

### 1. Download cuDNN
...

### 2. Configure PATH
...

[Complete guide with screenshots]
```

---

## Rollback Procedure

If CUDA causes issues:

### Quick Fallback to CPU:

```json
// config.json
{
  "detector": { "device": "cpu" },
  "stt": { 
    "providers": { 
      "whisper_local": { "device": "cpu" } 
    } 
  }
}
```

### Restore Old PyTorch:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

---

## Verification Checklist

Before moving to next phase:

- [ ] `nvidia-smi` shows GPU
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `torch.backends.cudnn.version()` returns number (e.g., 90100)
- [ ] No "cudnn_ops64_9.dll" errors in logs
- [ ] YOLO inference < 35ms average
- [ ] System runs without CUDA warnings

---

## Expected Performance Improvements

| Component | Before (CPU) | After (GPU) | Speedup |
|-----------|--------------|-------------|---------|
| YOLO | 85ms | 28ms | **3x** |
| Whisper | 10s | 2s | **5x** |
| VLM | 3-5s | 1-1.5s | **2-3x** |
| LLM | 0.7-2s | 0.5-1s | **1.4-2x** |
| **Overall E2E** | **15-20s** | **4-6s** | **3-4x** |

---

## Troubleshooting

### Issue: cuDNN still not found after installation

**Solution**:
```bash
# Verify DLL exists
dir "C:\Program Files\NVIDIA\CUDNN\v9.x\bin\cudnn_ops64_9.dll"

# If missing, re-download cuDNN
# Ensure you selected correct CUDA version
```

### Issue: PyTorch shows CUDA but cuDNN version is None

**Solution**:
```bash
# Rebuild PyTorch installation
pip cache purge
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Issue: GPU runs out of memory

**Solution**:
```python
# Reduce batch sizes or model sizes
# config.json
{
  "detector": { "active_model": "yolov8n" },  // Switch back
  "vlm": {
    "providers": {
      "huggingface": {
        "torch_dtype": "float16"  // Use FP16
      }
    }
  }
}
```

---

## Next Steps After Phase 2

Once CUDA is working and optimized:

1. ✅ **Benchmark current performance**
2. ✅ **Document setup process**
3. 🔄 **Consider VLM fine-tuning** (optional)
4. 🔄 **Add automated testing**

**DO NOT proceed with VLM fine-tuning until CUDA is fully operational.**
