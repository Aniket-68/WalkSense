import os
import yaml
import shutil
from pathlib import Path
from loguru import logger

def merge_yolo_datasets(base_data_dir, output_dir):
    """
    Merges multiple YOLO datasets into one unified dataset.
    """
    datasets = [
        "stair_up_downq.v1i.yolov11",
        "Gate.v1i.yolov11",
        "SpeedBump.v1i.yolov11",
        "manhole test.v1i.yolov11"
    ]
    
    # Target structure
    splits = ['train', 'valid']
    for split in splits:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    unified_classes = []
    class_map = {} # (dataset_name, original_id) -> unified_id

    # 1. Build unified class list
    global_class_id = 0
    for ds_name in datasets:
        ds_path = base_data_dir / ds_name
        with open(ds_path / "data.yaml", 'r') as f:
            config = yaml.safe_load(f)
            names = config['names']
            for i, name in enumerate(names):
                if name not in unified_classes:
                    unified_classes.append(name)
                    class_map[(ds_name, i)] = len(unified_classes) - 1
                else:
                    class_map[(ds_name, i)] = unified_classes.index(name)

    logger.info(f"Unified classes ({len(unified_classes)}): {unified_classes}")

    # 2. Copy and remap
    for ds_name in datasets:
        ds_path = base_data_dir / ds_name
        logger.info(f"Processing {ds_name}...")
        
        for split in splits:
            img_src = ds_path / split / 'images'
            lbl_src = ds_path / split / 'labels'
            
            if not img_src.exists():
                logger.warning(f"Skipping {split} for {ds_name} (images not found)")
                continue

            for img_file in img_src.glob("*.jpg"):
                # Copy image
                shutil.copy(img_file, output_dir / split / 'images' / f"{ds_name}_{img_file.name}")
                
                # Copy and remap label
                lbl_file = lbl_src / img_file.with_suffix('.txt').name
                if lbl_file.exists():
                    target_lbl = output_dir / split / 'labels' / f"{ds_name}_{lbl_file.name}"
                    with open(lbl_file, 'r') as f_in, open(target_lbl, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts: continue
                            orig_id = int(parts[0])
                            new_id = class_map[(ds_name, orig_id)]
                            f_out.write(f"{new_id} {' '.join(parts[1:])}\n")

    # 3. Create unified data.yaml
    unified_config = {
        'train': os.path.abspath(output_dir / 'train' / 'images'),
        'val': os.path.abspath(output_dir / 'valid' / 'images'), # Ultralytics often uses 'val'
        'nc': len(unified_classes),
        'names': unified_classes
    }
    
    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(unified_config, f)
    
    logger.success(f"Merged dataset created at {output_dir}")

if __name__ == "__main__":
    base_dir = Path("d:/Github/WalkSense/data/yolo/yolov11")
    out_dir = Path("d:/Github/WalkSense/Yolo/train_dataset") # Using a clean name
    merge_yolo_datasets(base_dir, out_dir)
