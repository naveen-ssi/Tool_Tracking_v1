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

# --- Directional Command Constants ---
COMMAND_ENUM = {
    "NEUTRAL": 0,
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4
}

FRAME_CENTER_X = 960 // 2  # 480
FRAME_CENTER_Y = 1080 // 2 # 540
COMMAND_THRESHOLD = 75  
# ----------------------------------------------

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
    
    stable_focus_point_960 = None
    realtime_midpoint_960 = None
    left_tip_960 = None
    right_tip_960 = None
    left_mask_960 = None
    right_mask_960 = None

    last_known_left_center = None
    last_known_right_center = None
    
    current_command_string = "NEUTRAL"
    last_sent_command_code = COMMAND_ENUM["NEUTRAL"]
    
    show_3d_mode = False 
    print(f"Starting live tracking on {device}... Press 'q' to quit, 'j' to toggle 2D/3D view.")

    while True:
        start_time = time.time()
        
        ret, full_frame = cap.read() 
        if not ret:
            print("Error: Could not read frame from camera. Exiting.")
            break

        frame = full_frame[0:height, 0:width//2]
        frame_right = full_frame[0:height, width//2:width]
        
        realtime_midpoint_960 = None
        current_command_string = "NEUTRAL" 
        
        results = model.predict(frame, device=device, verbose=False, imgsz=640, half=True) 

        detections_with_data = []
        for result in results:
            if result.masks is not None:
                for mask, box in zip(result.masks.xy, result.boxes):
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        box_coords = box.xyxy[0].cpu().numpy()
                        center_x = (box_coords[0] + box_coords[2]) / 2
                        center_y = (box_coords[1] + box_coords[3]) / 2
                        detections_with_data.append({
                            'mask': mask, 
                            'box': box_coords, 
                            'center': (center_x, center_y),
                            'size': (box_coords[2] - box_coords[0]) * (box_coords[3] - box_coords[1])
                        })
        
        left_instrument = None
        right_instrument = None
        
        # --- Tracking Logic (unchanged) ---
        if len(detections_with_data) < 2:
            last_known_left_center = None
            last_known_right_center = None
        
        elif last_known_left_center is None or last_known_right_center is None:
            detections_with_data.sort(key=lambda d: d['size'], reverse=True)
            top_two = detections_with_data[:2]
            
            if top_two[0]['center'][0] < top_two[1]['center'][0]:
                left_instrument = top_two[0]
                right_instrument = top_two[1]
            else:
                left_instrument = top_two[1]
                right_instrument = top_two[0]
            
            last_known_left_center = left_instrument['center']
            last_known_right_center = right_instrument['center']

        else:
            best_left_match = min(detections_with_data, 
                                  key=lambda det: np.linalg.norm(np.array(det['center']) - np.array(last_known_left_center)))
            left_instrument = best_left_match
            last_known_left_center = left_instrument['center'] 

            remaining_detections = [det for det in detections_with_data if det is not left_instrument]
            
            if remaining_detections:
                best_right_match = min(remaining_detections, 
                                       key=lambda det: np.linalg.norm(np.array(det['center']) - np.array(last_known_right_center)))
                right_instrument = best_right_match
                last_known_right_center = right_instrument['center']
            else:
                last_known_left_center = None
                last_known_right_center = None

        # --- End of Tracking Logic ---
        
        if left_instrument and right_instrument:
            left_mask_960 = left_instrument['mask']
            right_mask_960 = right_instrument['mask']
            left_tip_960, right_tip_960 = find_extreme_points_from_masks(left_mask_960, right_mask_960)

            realtime_midpoint_960 = (int((left_tip_960[0] + right_tip_960[0]) / 2), int((left_tip_960[1] + right_tip_960[1]) / 2))

            if stable_focus_point_960 is None:
                stable_focus_point_960 = realtime_midpoint_960
            else:
                dist = np.linalg.norm(np.array(stable_focus_point_960) - np.array(realtime_midpoint_960))
                if dist > STABLE_CIRCLE_RADIUS:
                    stable_x = int(stable_focus_point_960[0] * (1 - SMOOTHING_FACTOR) + realtime_midpoint_960[0] * SMOOTHING_FACTOR)
                    stable_y = int(stable_focus_point_960[1] * (1 - SMOOTHING_FACTOR) + realtime_midpoint_960[1] * SMOOTHING_FACTOR)
                    stable_focus_point_960 = (stable_x, stable_y)

            # --- <<< NEW: Command Logic (Dominant Axis) ---
            
            # Calculate deviation from the center
            dx = stable_focus_point_960[0] - FRAME_CENTER_X
            dy = stable_focus_point_960[1] - FRAME_CENTER_Y

            # Prioritize the dominant axis (is it more left/right or up/down?)
            if abs(dx) > abs(dy):
                # Horizontal movement is dominant
                if dx > COMMAND_THRESHOLD:
                    current_command_string = "RIGHT"
                elif dx < -COMMAND_THRESHOLD:
                    current_command_string = "LEFT"
                else:
                    current_command_string = "NEUTRAL"
            else:
                # Vertical movement is dominant
                if dy > COMMAND_THRESHOLD:
                    current_command_string = "DOWN"
                elif dy < -COMMAND_THRESHOLD:
                    current_command_string = "UP"
                else:
                    current_command_string = "NEUTRAL"
            # --- <<< END OF LOGIC ---

            current_command_code = COMMAND_ENUM[current_command_string]
            
            if current_command_code != last_sent_command_code:
                if serial_port and serial_port.is_open:
                    data_string = f"<{current_command_code}>\n"
                    
                    print(f"\nSending new command: {current_command_string} ({data_string.strip()})", flush=True)
                    
                    try:
                        serial_port.write(data_string.encode('utf-8'))
                        last_sent_command_code = current_command_code 
                    except serial.SerialException as e:
                        print(f"Error writing to serial port: {e}")
                        serial_port.close() 
                        serial_port = None 
            # ------------------------------------

        # --- OVERLAY AND DISPLAY LOGIC (Unchanged) ---
        
        if show_3d_mode:
            display_frame = cv2.hconcat([frame, frame_right])
            x_scale = 1.0 
            x_offset = 0  
        else:
            display_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            x_scale = 2.0 
            x_offset = 0  
            
        overlay_blue = display_frame.copy()
        overlay_green = display_frame.copy()
        overlay_yellow = display_frame.copy()

        PALE_BLUE = (255, 230, 204)
        YELLOW = (0, 255, 255)
        TRANSPARENT_GREEN = (0, 255, 0) 

        if realtime_midpoint_960 is not None:
            stable_focus_point_1920 = (int(stable_focus_point_960[0] * x_scale) + x_offset, stable_focus_point_960[1])
            realtime_midpoint_1920 = (int(realtime_midpoint_960[0] * x_scale) + x_offset, realtime_midpoint_960[1])
            left_tip_1920 = (int(left_tip_960[0] * x_scale) + x_offset, left_tip_960[1])
            right_tip_1920 = (int(right_tip_960[0] * x_scale) + x_offset, right_tip_960[1])

            left_mask_1920 = left_mask_960.copy().astype(np.float32)
            left_mask_1920[:, 0] = left_mask_1920[:, 0] * x_scale + x_offset
            
            right_mask_1920 = right_mask_960.copy().astype(np.float32)
            right_mask_1920[:, 0] = right_mask_1920[:, 0] * x_scale + x_offset

            cv2.fillPoly(overlay_green, [left_mask_1920.astype(np.int32)], TRANSPARENT_GREEN)
            cv2.fillPoly(overlay_green, [right_mask_1920.astype(np.int32)], TRANSPARENT_GREEN)
            display_frame = cv2.addWeighted(overlay_green, 0.3, display_frame, 0.7, 0)
            
            cv2.circle(overlay_blue, realtime_midpoint_1920, 60, PALE_BLUE, -1)
            display_frame = cv2.addWeighted(overlay_blue, 0.7, display_frame, 0.3, 0) 

            cv2.circle(overlay_yellow, stable_focus_point_1920, STABLE_CIRCLE_RADIUS, YELLOW, 3)
            display_frame = cv2.addWeighted(overlay_yellow, 0.5, display_frame, 0.5, 0)
            
            cv2.line(display_frame, left_tip_1920, right_tip_1920, PALE_BLUE, 2)
            cv2.circle(display_frame, left_tip_1920, 10, PALE_BLUE, -1)
            cv2.circle(display_frame, right_tip_1920, 10, PALE_BLUE, -1)
            
            text_pos = (stable_focus_point_1920[0] - 120, stable_focus_point_1920[1] - (STABLE_CIRCLE_RADIUS + 15))
            text_content = f"CAMERA FOCUS: {stable_focus_point_960}" 
            cv2.putText(display_frame, text_content, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # --- End of overlay logic ---

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"COMMAND: {current_command_string}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        print(f"Current Command: {current_command_string: <10}", end="\r", flush=True)
        
        cv2.imshow("AI Instrument Tracker", display_frame) 

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n'q' pressed. Exiting...") 
            break
        elif key == ord('j'): 
            show_3d_mode = not show_3d_mode 
            print(f"\nDisplay mode toggled. Show 3D: {show_3d_mode}")
        # ------------------------------------

    cap.release()
    cv2.destroyAllWindows()
    if serial_port and serial_port.is_open:
        serial_port.close()
        print("Serial port closed.")
    print("Application finished successfully.")

if __name__ == '__main__':
    main()