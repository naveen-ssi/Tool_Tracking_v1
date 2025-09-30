from roboflow import Roboflow
import os

# --- Configuration ---
# You need a valid Roboflow API key to download datasets.
# Get yours from your Roboflow account settings.
ROBOFLOW_API_KEY = "4wuQ1c8o9ykJW3ZOiHn7" 

# --- NEW DATASET INFORMATION ---
ROBOFLOW_WORKSPACE = "instrumentssurgical"
ROBOFLOW_PROJECT = "surginst-wdiex"
MODEL_VERSION = 2
# -----------------------------

def main():
    """
    Downloads the specified dataset from Roboflow.
    """
    print("Initializing Roboflow...")
    try:
        if "PASTE" in ROBOFLOW_API_KEY:
            print("Error: Please replace the placeholder with your actual Roboflow API key.")
            return

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(MODEL_VERSION)
        
        print(f"Downloading dataset '{ROBOFLOW_PROJECT}' version {version.version}...")
        
        # Download in yolov8 format, which is what our trainer needs
        version.download("yolov8")
        
        print("\n----------------------------------------------------")
        print("✅ Download complete!")
        print(f"The dataset is in a new folder named: '{ROBOFLOW_PROJECT}-{MODEL_VERSION}'")
        print("You are now ready to run the 'train_model.py' script.")
        print("----------------------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API key, workspace/project names, and internet connection.")

if __name__ == '__main__':
    main()

