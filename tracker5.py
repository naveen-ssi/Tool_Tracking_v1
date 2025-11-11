import cv2
import numpy as np
from ultralytics import YOLO
import torch # Import torch to check for GPU

# --- Configuration ---
# Point this to your new, single-class model file
MODEL_PATH = "SIGNET3.pt" 
VIDEO_SOURCE = 0 # Use 0 for your primary webcam
CONFIDENCE_THRESHOLD = 0.2 

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
    # Find the point with the largest x-value in the left mask
    left_tip = max(left_mask, key=lambda point: point[0])
    
    # Find the point with the smallest x-value in the right mask
    right_tip = min(right_mask, key=lambda point: point[0])
    
    return tuple(left_tip.astype(int)), tuple(right_tip.astype(int))

def main():
    """
    Main function to run the intelligent tip tracking application from a live webcam.
    """
    
    # --- Smart Device Selection ---
    # Check if a CUDA-enabled GPU is available, otherwise default to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("✅ PyTorch has successfully detected your NVIDIA GPU.")
        device = 0 # Use device index 0
    else:
        print("⚠️ WARNING: PyTorch could not detect a NVIDIA GPU.")
        print("Running on CPU. This will be very slow.")
        print("Ensure you have installed PyTorch with CUDA support.")
    # -----------------------------

    print(f"Loading single-class model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'. Is the webcam connected?")
        return
    
    # Set webcam resolution to 16:9 (1280x720)
    # Note: This is a request. The webcam will try to match it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Read back the resolution to see what we actually got
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution set to: {width}x{height}")
    
    stable_focus_point = None
    print(f"Starting live tracking on {device}... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Flip the frame horizontally (webcams are often mirrored)
        frame = cv2.flip(frame, 1)
        
        overlay = frame.copy() # Create an overlay for transparent drawing

        # Use the auto-detected device
        results = model.predict(frame, device=device, verbose=False) 

        # Process results to get masks
        detections = []
        for result in results:
            if result.masks is not None: # Check if the model provides masks
                for mask, box in zip(result.masks.xy, result.boxes):
                    if box.conf[0] > CONFIDENCE_THRESHOLD:
                        detections.append({'mask': mask, 'box': box.xyxy[0].cpu().numpy()})

        # Ensure we have at least two instruments to track
        if len(detections) >= 2:
            # Sort by the area of the bounding box to get the most prominent instruments
            detections.sort(key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]), reverse=True)
            top_two_detections = detections[:2]

            # --- Intelligent Left/Right Assignment ---
            det1, det2 = top_two_detections
            center_x1 = (det1['box'][0] + det1['box'][2]) / 2
            center_x2 = (det2['box'][0] + det2['box'][2]) / 2
            
            if center_x1 < center_x2:
                left_instrument, right_instrument = det1, det2
            else:
                left_instrument, right_instrument = det2, det1
            # ------------------------------------------

            # --- Find the true tips from the polygon masks ---
            left_mask = left_instrument['mask']
            right_mask = right_instrument['mask']
            left_tip, right_tip = find_extreme_points_from_masks(left_mask, right_mask)
            # -------------------------------------------------

            realtime_midpoint = (int((left_tip[0] + right_tip[0]) / 2), int((left_tip[1] + right_tip[1]) / 2))

            # --- Stabilization Logic (unchanged) ---
            if stable_focus_point is None:
                stable_focus_point = realtime_midpoint
            else:
                dist = np.linalg.norm(np.array(stable_focus_point) - np.array(realtime_midpoint))
                if dist > STABLE_CIRCLE_RADIUS:
                    stable_x = int(stable_focus_point[0] * (1 - SMOOTHING_FACTOR) + realtime_midpoint[0] * SMOOTHING_FACTOR)
                    stable_y = int(stable_focus_point[1] * (1 - SMOOTHING_FACTOR) + realtime_midpoint[1] * SMOOTHING_FACTOR)
                    stable_focus_point = (stable_x, stable_y)
            # --------------------------------------

            PALE_BLUE_FILLED = (255, 230, 204) # BGR format
            YELLOW_OUTLINE = (0, 255, 255)
            MASK_FILL_COLOR = (0, 150, 255) # A distinct color for masks

            # --- Visualization ---
            # Draw filled transparent masks
            # Create a black mask image to draw the polygons on
            mask_img = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask_img, [left_mask.astype(np.int32)], MASK_FILL_COLOR)
            cv2.fillPoly(mask_img, [right_mask.astype(np.int32)], MASK_FILL_COLOR)
            
            # Blend the mask onto the overlay (adjust alpha for desired transparency)
            alpha = 0.3 # 30% opaque
            cv2.addWeighted(mask_img, alpha, overlay, 1 - alpha, 0, overlay)

            # Draw the real-time midpoint (smaller, filled circle, 30% transparent)
            cv2.circle(overlay, realtime_midpoint, 60, PALE_BLUE_FILLED, -1)
            
            # Blend the overlay (with filled masks and inner circle) onto the frame
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0) # This blending now also includes the masks

            # Draw the stable focus point (larger, yellow outline circle)
            cv2.circle(frame, stable_focus_point, STABLE_CIRCLE_RADIUS, YELLOW_OUTLINE, 3)
            
            # Draw line between tips
            cv2.line(frame, left_tip, right_tip, PALE_BLUE_FILLED, 2)
            
            # Draw solid circles on the tips
            cv2.circle(frame, left_tip, 10, PALE_BLUE_FILLED, -1) 
            cv2.circle(frame, right_tip, 10, PALE_BLUE_FILLED, -1) 

            # Display "ROBOT FOCUS" text
            cv2.putText(frame, f"ROBOT FOCUS: {stable_focus_point}", (stable_focus_point[0] - 120, stable_focus_point[1] - (STABLE_CIRCLE_RADIUS + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("AI Instrument Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished successfully.")

if __name__ == '__main__':
    main()

