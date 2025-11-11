import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os     # NEW IMPORT for finding paths
import sys    # NEW IMPORT for finding paths
import multiprocessing # NEW IMPORT for .exe compatibility
import time   # NEW IMPORT for FPS counter

# --- Configuration ---
# This function is CRITICAL for the .exe to find your files
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # This is the path to the file inside the bundled .exe
        base_path = sys._MEIPASS
    except Exception:
        # This is the path for running normally (not as an .exe)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Use the new function to find our bundled model file
MODEL_PATH = resource_path("SIGNET4.pt")

# UPDATED: Set VIDEO_SOURCE to 0 to use the default live camera/capture card
VIDEO_SOURCE = 0 
CONFIDENCE_THRESHOLD = 0.5 

# --- Stabilization Parameters ---
STABLE_CIRCLE_RADIUS = 180 
SMOOTHING_FACTOR = 0.05    
# ------------------------------------

def find_extreme_points_from_masks(left_mask, right_mask):
    """
    Finds the innermost point (tip) for each instrument's polygon mask.
    - For the left instrument, it's the point with the maximum X-coordinate.
    - For the right instrument, it's the point with the minimum X-coordinate.
    """
    left_tip = max(left_mask, key=lambda point: point[0])
    right_tip = min(right_mask, key=lambda point: point[0])
    return tuple(left_tip.astype(int)), tuple(right_tip.astype(int))

def main():
    # This function call is CRITICAL for the .exe to run
    # It must be the very first line inside main()
    multiprocessing.freeze_support()

    # --- Smart Device Selection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("✅ PyTorch has successfully detected your NVIDIA GPU.")
        device = 0 # Use device index 0
    else:
        print("⚠️ WARNING: PyTorch could not detect a NVIDIA GPU. Running on CPU.")
    # -----------------------------

    print(f"Loading single-class model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        print("Please ensure your endoscope/webcam is connected.")
        # Pause to allow user to read the error in the console
        input("Press Enter to exit...") 
        return
        
    # --- NEW: Request 1080p Resolution ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested 1920x1080, camera provides: {width}x{height}")
    # -------------------------------------

    # --- Video Recording Setup REMOVED ---
    
    stable_focus_point = None
    print(f"Starting live tracking on {device}... Press 'q' to quit.")

    while True:
        # --- NEW: FPS Counter Start ---
        start_time = time.time()
        # ----------------------------
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera. Exiting.")
            break

        # Create overlays for transparent drawing
        overlay_blue = frame.copy()
        overlay_green = frame.copy()
        overlay_yellow = frame.copy()

        # --- OPTIMIZED: Added imgsz=640 and half=True for massive speedup ---
        results = model.predict(frame, device=device, verbose=False, imgsz=640, half=True) 

        detections = []
        for result in results:
            if result.masks is not None:
                for mask, box in zip(result.masks.xy, result.boxes):
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        # We need to scale the mask coordinates up if we used a smaller imgsz
                        # This isn't needed for masks.xy, as they are already in original image coordinates.
                        detections.append({'mask': mask, 'box': box.xyxy[0].cpu().numpy()})

        if len(detections) >= 2:
            detections.sort(key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]), reverse=True)
            top_two_detections = detections[:2]

            det1, det2 = top_two_detections
            center_x1 = (det1['box'][0] + det1['box'][2]) / 2
            center_x2 = (det2['box'][0] + det2['box'][2]) / 2
            
            if center_x1 < center_x2:
                left_instrument, right_instrument = det1, det2
            else:
                left_instrument, right_instrument = det2, det1

            left_mask = left_instrument['mask']
            right_mask = right_instrument['mask']
            left_tip, right_tip = find_extreme_points_from_masks(left_mask, right_mask)

            realtime_midpoint = (int((left_tip[0] + right_tip[0]) / 2), int((left_tip[1] + right_tip[1]) / 2))

            if stable_focus_point is None:
                stable_focus_point = realtime_midpoint
            else:
                dist = np.linalg.norm(np.array(stable_focus_point) - np.array(realtime_midpoint))
                if dist > STABLE_CIRCLE_RADIUS:
                    stable_x = int(stable_focus_point[0] * (1 - SMOOTHING_FACTOR) + realtime_midpoint[0] * SMOOTHING_FACTOR)
                    stable_y = int(stable_focus_point[1] * (1 - SMOOTHING_FACTOR) + realtime_midpoint[1] * SMOOTHING_FACTOR)
                    stable_focus_point = (stable_x, stable_y)

            PALE_BLUE = (255, 230, 204)
            YELLOW = (0, 255, 255)
            # --- NEW: Transparent Green Color ---
            TRANSPARENT_GREEN = (0, 255, 0) 

            # --- NEW VISUALIZATION LOGIC ---
            # 1. Draw transparent filled green masks (30% opaque)
            cv2.fillPoly(overlay_green, [left_mask.astype(np.int32)], TRANSPARENT_GREEN)
            cv2.fillPoly(overlay_green, [right_mask.astype(np.int32)], TRANSPARENT_GREEN)
            frame = cv2.addWeighted(overlay_green, 0.7, frame, 0.7, 0)
            
            # 2. Draw transparent inner blue circle (70% opaque)
            cv2.circle(overlay_blue, realtime_midpoint, 60, PALE_BLUE, -1)
            frame = cv2.addWeighted(overlay_blue, 0.7, frame, 0.3, 0) 

            # 3. Draw transparent outer yellow circle (50% opaque)
            cv2.circle(overlay_yellow, stable_focus_point, STABLE_CIRCLE_RADIUS, YELLOW, 3)
            frame = cv2.addWeighted(overlay_yellow, 0.5, frame, 0.5, 0)
            
            # 4. Draw opaque lines and text on top of the blended image
            cv2.line(frame, left_tip, right_tip, PALE_BLUE, 2)
            cv2.circle(frame, left_tip, 10, PALE_BLUE, -1)
            cv2.circle(frame, right_tip, 10, PALE_BLUE, -1)
            cv2.putText(frame, f"CAMERA FOCUS: {stable_focus_point}", (stable_focus_point[0] - 120, stable_focus_point[1] - (STABLE_CIRCLE_RADIUS + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # -------------------------------

        # --- NEW: FPS Counter Display ---
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # ------------------------------

        cv2.imshow("AI Instrument Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application finished successfully.")

# This "if" block is CRITICAL for the .exe to work
if __name__ == '__main__':
    main()