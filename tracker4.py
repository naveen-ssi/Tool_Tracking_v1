import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# Point this to your new, single-class model file
MODEL_PATH = "singleclass_seg.pt" 
VIDEO_SOURCE = "BIMA MANTRA.mp4" # Or use 0 for your webcam
CONFIDENCE_THRESHOLD = 0.7 

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
    Main function to run the intelligent tip tracking application using polygon masks.
    """
    print(f"Loading single-class model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        return

    # --- Video Recording Setup ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_filename = 'intelligent_tip_tracker_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be recorded to '{output_filename}'")
    
    stable_focus_point = None
    print("Starting live tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        # Setting retopo=True might be needed for some models to get clean masks
        results = model.predict(frame, device=0, verbose=False) 

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

            PALE_BLUE = (255, 230, 204)
            YELLOW = (0, 255, 255)
            MASK_COLOR = (120, 120, 120)

            # --- Visualization ---
            # Draw the actual polygon masks detected by the AI
            cv2.polylines(frame, [left_mask.astype(np.int32)], isClosed=True, color=MASK_COLOR, thickness=2)
            cv2.polylines(frame, [right_mask.astype(np.int32)], isClosed=True, color=MASK_COLOR, thickness=2)
            
            # Draw the transparent circles and line based on the inferred TIPS
            cv2.circle(overlay, realtime_midpoint, 60, PALE_BLUE, -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0) # 30% transparent

            cv2.circle(frame, stable_focus_point, STABLE_CIRCLE_RADIUS, YELLOW, 3)
            cv2.line(frame, left_tip, right_tip, PALE_BLUE, 2)
            cv2.circle(frame, left_tip, 10, PALE_BLUE, -1) # Solid circle on the tip
            cv2.circle(frame, right_tip, 10, PALE_BLUE, -1) # Solid circle on the tip

            cv2.putText(frame, f"ROBOT FOCUS: {stable_focus_point}", (stable_focus_point[0] - 120, stable_focus_point[1] - (STABLE_CIRCLE_RADIUS + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("AI Instrument Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Application finished successfully.")

if __name__ == '__main__':
    main()

