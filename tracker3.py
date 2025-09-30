import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# IMPORTANT: Make sure this is the exact name of your custom-trained model file.
MODEL_PATH = "singleclass_seg.pt" 
VIDEO_SOURCE = "Z:\27. Naveen Kumar\SSi Maya Shared\Surgery Videos for Model Training\PT 2 Cholecystectomy_15122020.mp4" # Or use 0 for your webcam
CONFIDENCE_THRESHOLD = 0.1 # Start at 50% and adjust as needed.

# --- New Stabilization Parameters ---
STABLE_CIRCLE_RADIUS = 180 # 3x the inner circle's radius
SMOOTHING_FACTOR = 0.05    # Controls how fast the stable circle moves. Lower is smoother.
# ------------------------------------

def get_enclosing_circle(box):
    """Calculates the center and radius of a circle that encloses the bounding box."""
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    radius = int(np.sqrt(((x2 - center_x) ** 2) + ((y2 - center_y) ** 2)))
    return (center_x, center_y), radius

def main():
    """
    Main function to run the live instrument tracking application with stabilization.
    """
    print(f"Loading your custom-trained model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure '{MODEL_PATH}' is in the same folder as this script.")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        return

    # --- Video Recording Setup ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_filename = 'tracked_output_stabilized.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be recorded to '{output_filename}'")
    
    # Get the class names from the trained model
    class_names = model.names
    print(f"Model classes: {class_names}")
    
    # --- State variable for the stabilized focus point ---
    stable_focus_point = None

    print("Starting live tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Create copies of the frame for drawing transparent overlays
        overlay_blue = frame.copy()
        overlay_yellow = frame.copy()

        results = model.predict(frame, device=0, verbose=False) # Use GPU

        # Process results to get detections with class names
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls[0])
                    detections.append({
                        "box": box.xyxy[0].cpu().numpy(),
                        "class_name": class_names.get(class_id, "Unknown"),
                        "confidence": box.conf[0].cpu().numpy()
                    })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        top_two_detections = detections[:2]

        if len(top_two_detections) == 2:
            det1, det2 = top_two_detections

            center1, radius1 = get_enclosing_circle(det1['box'])
            center2, radius2 = get_enclosing_circle(det2['box'])

            # Determine left/right for consistent labeling
            if center1[0] < center2[0]:
                left_center, right_center, left_radius, right_radius, left_name, right_name = center1, center2, radius1, radius2, det1['class_name'], det2['class_name']
            else:
                left_center, right_center, left_radius, right_radius, left_name, right_name = center2, center1, radius2, radius1, det2['class_name'], det1['class_name']

            realtime_midpoint = (int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2))

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

            # --- Draw Transparent Visualizations ---
            # 1. Draw 30% transparent blue elements on their own overlay
            cv2.circle(overlay_blue, realtime_midpoint, 60, PALE_BLUE, -1) # Inner midpoint remains FILLED
            cv2.circle(overlay_blue, left_center, left_radius, PALE_BLUE, 3) # Instrument circle is now an OUTLINE
            cv2.circle(overlay_blue, right_center, right_radius, PALE_BLUE, 3) # Instrument circle is now an OUTLINE
            frame = cv2.addWeighted(overlay_blue, 0.7, frame, 0.3, 0) # 30% transparent (alpha=0.7)

            # 2. Draw 50% transparent yellow element on its own overlay
            cv2.circle(overlay_yellow, stable_focus_point, STABLE_CIRCLE_RADIUS, YELLOW, 3) # Stable circle is now an OUTLINE
            frame = cv2.addWeighted(overlay_yellow, 0.5, frame, 0.5, 0) # 50% transparent (alpha=0.5)

            # --- Draw Opaque Visualizations on the final blended frame ---
            cv2.line(frame, left_center, right_center, PALE_BLUE, 2)
            cv2.putText(frame, left_name, (left_center[0] - 40, left_center[1] - left_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, right_name, (right_center[0] - 40, right_center[1] - right_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Endoscope Focus: {stable_focus_point}", (stable_focus_point[0] - 120, stable_focus_point[1] - (STABLE_CIRCLE_RADIUS + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

