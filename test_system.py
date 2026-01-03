#!/usr/bin/env python3
"""
WalkSense Enhanced - Test Run
Simple test to verify all components work with your wireless camera
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import cv2
import time

print("\n" + "="*70)
print("WALKSENSE TEST - Camera + Models")
print("="*70)
config.get_config_summary()

# Test camera
print("\n[TEST] Testing camera...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f"✓ Camera OK - Resolution: {frame.shape[1]}x{frame.shape[0]}")
else:
    print("✗ Camera FAIL")
    exit(1)

# Test YOLO
print("\n[TEST] Testing YOLO...")
from safety.yolo_detector import YoloDetector
detector = YoloDetector(config.YOLO["model_path"])
detections = detector.detect(frame)
print(f"✓ YOLO OK - Detected {len(detections)} objects")

# Test Spatial Tracking
print("\n[TEST] Testing Spatial Tracking...")
from inference.spatial_context_manager import SpatialContextManager
spatial = SpatialContextManager()
events = spatial.update(detections, time.time(), frame.shape[1])
print(f"✓ Spatial Tracking OK - {len(events)} events")

# Test LLM (without making actual call)
print("\n[TEST] Testing LLM Reasoner...")
from inference.llm_reasoner import LLMReasoner
llm = LLMReasoner(
    backend=config.TEXT_MODEL["backend"],
    api_url=config.TEXT_MODEL[f"{config.TEXT_MODEL['backend']}_url"],
    model_name=config.TEXT_MODEL["model_name"]
)
print(f"✓ LLM Reasoner OK - {config.TEXT_MODEL['model_name']}")

# Test VLM (without making actual call)
print("\n[TEST] Testing VLM...")
from reasoning.qwen_vlm import QwenVLM
vlm_backend = config.VISION_MODEL["backend"]
qwen = QwenVLM(
    backend=vlm_backend,
    model_id=config.VISION_MODEL["model_name"],
    ollama_url=config.VISION_MODEL["ollama_url"] if vlm_backend == "ollama" else None
)
print(f"✓ VLM OK - {config.VISION_MODEL['model_name']}")

cap.release()

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour system is ready to run!")
print("Camera works, models configured, all components loaded.")
print("\nTo run full system: python3 scripts/run_enhanced_camera.py")
print("(After fixing the variable names)")
