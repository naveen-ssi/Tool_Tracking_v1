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

# --- Calibration Data ---
MM_PER_PIXEL = 0.073 

# --- Center Point ---
FRAME_CENTER_X = 960 // 2  # 480
FRAME_CENTER_Y = 1080 // 2 # 540

# --- Encoding Logic ---
def int8_2_byte(n):
    """Encodes an integer (-128 to 127) into a single byte."""
    n = int(n)
    n = max(min(n, 127), -128)
    if(n < 0):
        n = 127 - n
        return min(n, 255)
    else:
        return min(n, 127)

def find_extreme_points_from_masks(left_mask, right_mask):
    left_tip = max(left_mask, key=lambda point: point[0])
    right_tip = min(right_mask, key=lambda point: point[0])
    return tuple(left_tip.astype(int)), tuple(right_tip.astype(int))

def main():
    multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("PyTorch has successfully detected your NVIDIA GPU.")
        device = 0 
    else:
        print("WARNING: PyTorch could not detect a NVIDIA GPU. Running on CPU.")
    
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
        input("Press Enter to exit...") 
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    stable_focus_point_960 = None
    realtime_midpoint_960 = None
    last_known_left_center = None
    last_known_right_center = None
    
    show_3d_mode = False 
    
    print(f"Starting tracking... Press 'q' to quit, 'j' to toggle 2D/3D view.")

    while True:
        start_time = time.time()
        
        ret, full_frame = cap.read() 
        if not ret: break

        frame = full_frame[0:height, 0:width//2]
        frame_right = full_frame[0:height, width//2:width]
        
        realtime_midpoint_960 = None
        
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
        
        # --- Tracking Logic ---
        left_instrument = None
        right_instrument = None
        
        if len(detections_with_data) < 2:
            last_known_left_center = None; last_known_right_center = None
        elif last_known_left_center is None or last_known_right_center is None:
            detections_with_data.sort(key=lambda d: d['size'], reverse=True)
            top_two = detections_with_data[:2]
            if top_two[0]['center'][0] < top_two[1]['center'][0]:
                left_instrument = top_two[0]; right_instrument = top_two[1]
            else:
                left_instrument = top_two[1]; right_instrument = top_two[0]
            last_known_left_center = left_instrument['center']; last_known_right_center = right_instrument['center']
        else:
            best_left_match = min(detections_with_data, key=lambda det: np.linalg.norm(np.array(det['center']) - np.array(last_known_left_center)))
            left_instrument = best_left_match
            last_known_left_center = left_instrument['center'] 
            remaining_detections = [det for det in detections_with_data if det is not left_instrument]
            if remaining_detections:
                best_right_match = min(remaining_detections, key=lambda det: np.linalg.norm(np.array(det['center']) - np.array(last_known_right_center)))
                right_instrument = best_right_match
                last_known_right_center = right_instrument['center']
            else:
                last_known_left_center = None; last_known_right_center = None
        
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

            # --- Calculate Deltas ---
            dx_pixel = stable_focus_point_960[0] - FRAME_CENTER_X
            dy_pixel = stable_focus_point_960[1] - FRAME_CENTER_Y
            
            dx_mm = dx_pixel * MM_PER_PIXEL
            dy_mm = dy_pixel * MM_PER_PIXEL
            dz_mm = 0.0 

            # --- Send Binary Packet ---
            if serial_port and serial_port.is_open:
                val_dx = int(dx_mm)
                val_dy = int(dy_mm)
                val_dz = int(dz_mm)

                # Encode first so we can see the values in the print
                enc_dx = int8_2_byte(val_dx)
                enc_dy = int8_2_byte(val_dy)
                enc_dz = int8_2_byte(val_dz)

                tx_pkt = [
                    0x53, 
                    enc_dx, 
                    enc_dy, 
                    enc_dz, 
                    0x45, 
                    0x0D, 
                    0x0A
                ]

                try:
                    serial_port.write(bytes(tx_pkt))
                    
                    # --- <<< UPDATED PRINT LOGIC >>> ---
                    # 1. Hex String (The Packet)
                    hex_str = ' '.join(f'{b:02X}' for b in tx_pkt)
                    
                    # 2. Encoded Values (The 0-255 ints)
                    enc_str = f"{enc_dx:03d}, {enc_dy:03d}, {enc_dz:03d}"
                    
                    # 3. Real Values (The mm error)
                    real_str = f"{val_dx: >3}, {val_dy: >3}"

                    print(f"TX: {hex_str} | Encoded: {enc_str} | Real(mm): {real_str}", end="\r", flush=True)
                    # -----------------------------------
                    
                except serial.SerialException as e:
                    print(f"\nError writing to serial port: {e}")
                    serial_port.close() 
                    serial_port = None 

        # --- Display Logic ---
        if show_3d_mode:
            display_frame = cv2.hconcat([frame, frame_right])
            x_scale = 1.0; x_offset = 0  
        else:
            display_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            x_scale = 2.0; x_offset = 0  
            
        overlay_blue = display_frame.copy()
        overlay_yellow = display_frame.copy()
        PALE_BLUE = (255, 230, 204)
        YELLOW = (0, 255, 255)

        if realtime_midpoint_960 is not None:
            stable_focus_point_1920 = (int(stable_focus_point_960[0] * x_scale) + x_offset, stable_focus_point_960[1])
            realtime_midpoint_1920 = (int(realtime_midpoint_960[0] * x_scale) + x_offset, realtime_midpoint_960[1])
            left_tip_1920 = (int(left_tip_960[0] * x_scale) + x_offset, left_tip_960[1])
            right_tip_1920 = (int(right_tip_960[0] * x_scale) + x_offset, right_tip_960[1])

            cv2.circle(overlay_blue, realtime_midpoint_1920, 60, PALE_BLUE, -1)
            display_frame = cv2.addWeighted(overlay_blue, 0.7, display_frame, 0.3, 0) 
            cv2.circle(overlay_yellow, stable_focus_point_1920, STABLE_CIRCLE_RADIUS, YELLOW, 3)
            display_frame = cv2.addWeighted(overlay_yellow, 0.5, display_frame, 0.5, 0)
            cv2.line(display_frame, left_tip_1920, right_tip_1920, PALE_BLUE, 2)
            
            text_pos = (stable_focus_point_1920[0] - 120, stable_focus_point_1920[1] - (STABLE_CIRCLE_RADIUS + 15))
            cv2.putText(display_frame, f"FOCUS: {stable_focus_point_960}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        fps = 1 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("AI Instrument Tracker", display_frame) 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('j'): show_3d_mode = not show_3d_mode 

    cap.release()
    cv2.destroyAllWindows()
    if serial_port and serial_port.is_open: serial_port.close()

if __name__ == '__main__':
    main()