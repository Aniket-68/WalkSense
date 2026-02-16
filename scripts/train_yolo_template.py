import os
import argparse
from ultralytics import YOLO
from loguru import logger

def train_custom_yolo(data_yaml, base_model="yolo11m.pt", epochs=50, imgsz=640, project="WalkSense_Stairs"):
    """
    Fine-tunes YOLO on custom data.
    
    Args:
        data_yaml (str): Path to the dataset yaml file (Roboflow exports this).
        base_model (str): The starting weights (yolo11m.pt recommended for balance).
        epochs (int): Number of training epochs.
        imgsz (int): Image size for training.
        project (str): Directory name for saving results.
    """
    if not os.path.exists(data_yaml):
        logger.error(f"Data YAML not found at {data_yaml}. Please ensure you have labeled your data and exported it.")
        return

    logger.info(f"Starting training with base model: {base_model}")
    logger.info(f"Dataset: {data_yaml}")
    
    # Load the model
    model = YOLO(base_model)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        name="tune_v1",
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu" # Auto-detect GPU
    )
    
    logger.success(f"Training complete. Results saved to {project}/tune_v1")
    logger.info("The best weights will be at: " + os.path.join(project, "tune_v1", "weights", "best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for WalkSense")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model weights")
    
    args = parser.parse_args()
    
    train_custom_yolo(args.data, base_model=args.model, epochs=args.epochs)
