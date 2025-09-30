from ultralytics import YOLO
import os
import yaml
import torch # Import the torch library

# --- Configuration ---
# This must match the dataset folder downloaded by the previous script.
DATASET_FOLDER_NAME = "surginst-2" 
# --------------------

def fix_yaml_paths(yaml_path):
    """
    Overwrites the paths in data.yaml to be correct and absolute.
    This is the definitive fix that prevents path-related errors during training.
    """
    print("Applying definitive path correction to data.yaml...")

    # The absolute path to the folder containing the YAML file.
    dataset_root_path = os.path.abspath(os.path.dirname(yaml_path))

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # CRITICAL FIX: Force the paths to the correct, known structure.
    data['path'] = dataset_root_path
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    
    # Check if 'test' exists in the dataset before trying to set its path.
    if 'test' in data:
        data['test'] = 'test/images'
        print(f"Forced 'test' to: {data['test']}")

    print(f"Set 'path' to: {data['path']}")
    print(f"Forced 'train' to: {data['train']}")
    print(f"Forced 'val' to: {data['val']}")

    # Overwrite the yaml file with the corrected configuration.
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("Path configuration complete. The YAML file has been rewritten correctly.")


def main():
    """
    This script trains a YOLOv8 model on our downloaded dataset.
    """
    # --- Pre-flight check for GPU availability ---
    if not torch.cuda.is_available():
        print("\n" + "="*50)
        print("❌ ERROR: PyTorch cannot detect a compatible NVIDIA GPU.")
        # (Error message omitted for brevity as the check is now passing)
        return # Stop the script
    # --------------------------------------------------

    print("✅ PyTorch has successfully detected your NVIDIA GPU.")
    print("Starting the model training process...")

    # Step 1: Find the 'data.yaml' file.
    data_yaml_path = os.path.join(DATASET_FOLDER_NAME, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        print(f"Error: Could not find 'data.yaml' inside the '{DATASET_FOLDER_NAME}' folder.")
        return

    fix_yaml_paths(data_yaml_path)
    
    print(f"\nFound and correctly configured dataset at: {data_yaml_path}")

    # Step 2: Load a pre-trained YOLOv8 model.
    model = YOLO('yolov8n.pt')
    print("Loaded pre-trained YOLOv8 model.")

    # Step 3: Train the model!
    print("\nStarting training... This will now run on the GPU and should be much faster.")
    
    results = model.train(
        data=data_yaml_path, 
        epochs=50, 
        imgsz=640,
        device=0, 
        workers=0 # Prevents Windows paging file error.
        # 'resume=True' has been removed to ensure a clean start.
    )

    print("\n----------------------------------------------------")
    print("✅ Training complete!")
    print("The trained model is 'best.pt' and is located inside the 'runs/detect/train/weights' folder.")
    print("----------------------------------------------------")

if __name__ == '__main__':
    main()

