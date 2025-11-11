import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import sys
import multiprocessing
import time
import serial                 
import serial.tools.list_ports

# --- Configuration ---
def resource_path(relative_path):
    
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

MODEL_PATH = resource_path("SIGNET4.pt")

VIDEO_SOURCE = 0 
CONFIDENCE_THRESHOLD = 0.5 

# --- Stabilization Parameters ---
STABLE_CIRCLE_RADIUS = 180 
SMOOTHING_FACTOR = 0.05    

# --- Serial Port Configuration ---
SERIAL_ENABLED = True      
SERIAL_PORT = "COM3"       
BAUD_RATE = 115200         
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
    multiprocessing.freeze_support()

    # --- Smart Device Selection ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("PyTorch has successfully detected your NVIDIA GPU.")
        device = 0 # Use device index 0
    else:
        print("WARNING: PyTorch could not detect a NVIDIA GPU. Running on CPU.")
    # -----------------------------

    # --- Serial Port Initialization ---
    serial_port = None
    if SERIAL_ENABLED:
        try:
            print(f"Attempting to connect to serial port {SERIAL_PORT} at {BAUD_RATE}...")
            serial_port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print("Serial port connected successfully.")
        except serial.SerialException as e:
            print(f"WARNING: Could not open serial port {SERIAL_PORT}: {e}")
            print("Available ports:")
            ports = serial.tools.list_ports.comports()
            for port, desc, hwid in sorted(ports):
                print(f"- {port}: {desc} [{hwid}]")
            print("Serial output will be disabled.")
    # -------------------------------------

    print(f"Loading single-class model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        print("Please ensure your endoscope/webcam is connected.")
        input("Press Enter to exit...") 
        return
        
    # --- Request 1080p Resolution ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested 1920x1080, camera provides: {width}x{height}")
    # -------------------------------------
    
    stable_focus_point = None
    show_3d_mode = False # <<< NEW: State variable for 2D/3D toggle
    print(f"Starting live tracking on {device}... Press 'q' to quit, 'j' to toggle 2D/3D view.")

    while True:
        start_time = time.time()
        
        ret, full_frame = cap.read() # Read the full 1920x1080 frame
        if not ret:
            print("Error: Could not read frame from camera. Exiting.")
            break

        # --- Crop to Left Half (960x1080) for processing ---
        # 'frame' is the processed left half
        frame = full_frame[0:height, 0:width//2]
        # ---------------------------------------------------------

        # Create overlays for transparent drawing (now based on the 960x1080 frame)
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
            TRANSPARENT_GREEN = (0, 255, 0) 

            # --- VISUALIZATION LOGIC ---
            cv2.fillPoly(overlay_green, [left_mask.astype(np.int32)], TRANSPARENT_GREEN)
            cv2.fillPoly(overlay_green, [right_mask.astype(np.int32)], TRANSPARENT_GREEN)
            frame = cv2.addWeighted(overlay_green, 0.3, frame, 0.7, 0)
            
            cv2.circle(overlay_blue, realtime_midpoint, 60, PALE_BLUE, -1)
            frame = cv2.addWeighted(overlay_blue, 0.7, frame, 0.3, 0) 

            cv2.circle(overlay_yellow, stable_focus_point, STABLE_CIRCLE_RADIUS, YELLOW, 3)
            frame = cv2.addWeighted(overlay_yellow, 0.5, frame, 0.5, 0)
            
            cv2.line(frame, left_tip, right_tip, PALE_BLUE, 2)
            cv2.circle(frame, left_tip, 10, PALE_BLUE, -1)
            cv2.circle(frame, right_tip, 10, PALE_BLUE, -1)
            cv2.putText(frame, f"CAMERA FOCUS: {stable_focus_point}", (stable_focus_point[0] - 120, stable_focus_point[1] - (STABLE_CIRCLE_RADIUS + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # -------------------------------

            # --- Send coordinates to Serial Port ---
            if serial_port and serial_port.is_open:
                data_string = f"<{stable_focus_point[0]},{stable_focus_point[1]}>\n"
                print(f"Sending to {SERIAL_PORT}: {data_string.strip()}")
                try:
                    serial_port.write(data_string.encode('utf-8'))
                except serial.SerialException as e:
                    print(f"Error writing to serial port: {e}")
                    serial_port.close() 
                    serial_port = None 
            # ------------------------------------------

        # --- FPS Counter Display ---
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # --- NEW: Display Mode Toggle Logic ---
        if show_3d_mode:
            # 3D Mode: Show processed left half + original right half
            # Get the original right half from the full_frame
            right_half = full_frame[0:height, width//2:width]
            
            # Combine the processed left 'frame' with the unprocessed 'right_half'
            display_frame = cv2.hconcat([frame, right_half])
        else:
            # 2D Mode: Resize the processed left frame to fill the screen
            display_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        # -------------------------------------------

        cv2.imshow("AI Instrument Tracker", display_frame) # Show the final composited frame

        # --- MODIFIED: Keypress Handling ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("'q' pressed. Exiting...")
            break
        elif key == ord('j'): # <<< NEW
            show_3d_mode = not show_3d_mode # Toggle the mode
            print(f"Display mode toggled. Show 3D: {show_3d_mode}")
        # ------------------------------------

    cap.release()
    cv2.destroyAllWindows()
    if serial_port and serial_port.is_open:
        serial_port.close()
        print("Serial port closed.")
    print("Application finished successfully.")

if __name__ == '__main__':
    main()