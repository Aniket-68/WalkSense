import os

# ---------- IMPORTANT MEMORY FIX ----------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable MLflow completely
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_EXPERIMENT_ID"] = ""
os.environ["MLFLOW_RUN_ID"] = ""

from ultralytics import YOLO
import torch
import sys


def train_model():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo11m.pt")

    results = model.train(
        data=r"d:\Github\WalkSense\Yolo\train_dataset\data.yaml",
        epochs=50,
        imgsz=640,
        device=0,

        # -------- MEMORY SAFE SETTINGS --------
        batch=8,              # IMPORTANT (reduce from auto)
        amp=True,             # mixed precision (huge memory saving)
        cache=False,          # prevents VRAM cache explosion
        workers=4,

        project=r"d:\Github\WalkSense\models\yolo",
        name="train_stairs_v1",
        exist_ok=True,
    )


if __name__ == '__main__':
    train_model()
