import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# IMPORTANT: Make sure this is the exact name of your custom-trained model file.
MODEL_PATH = "my_custom_model.pt" 
VIDEO_SOURCE = "demoreel.mp4" # Or use 0 for your webcam
CONFIDENCE_THRESHOLD = 0.5 # Start at 50% and adjust as needed.
# --------------------

def get_enclosing_circle(box):
    """Calculates the center and radius of a circle that encloses the bounding box."""
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    # Radius is the distance from the center to any corner
    radius = int(np.sqrt(((x2 - center_x) ** 2) + ((y2 - center_y) ** 2)))
    return (center_x, center_y), radius

def main():
    """
    Main function to run the live instrument tracking application with new visualization.
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
    output_filename = 'tracked_output_circles.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    print(f"Output will be recorded to '{output_filename}'")
    # -----------------------------

    print("Starting live tracking... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Create a copy for the transparent overlay
        overlay = frame.copy()

        # Run the AI model on the frame
        results = model.predict(frame, device=0, verbose=False) # Use GPU

        # Process results
        detections = [box for result in results for box in result.boxes if box.conf[0] > CONFIDENCE_THRESHOLD]
        detections.sort(key=lambda b: b.conf[0], reverse=True)
        top_two_detections = detections[:2]

        if len(top_two_detections) == 2:
            box1_data = top_two_detections[0].xyxy[0].cpu().numpy()
            box2_data = top_two_detections[1].xyxy[0].cpu().numpy()

            center1, radius1 = get_enclosing_circle(box1_data)
            center2, radius2 = get_enclosing_circle(box2_data)

            # Determine left and right for consistent line drawing
            left_center, right_center = (center1, center2) if center1[0] < center2[0] else (center2, center1)
            left_radius, right_radius = (radius1, radius2) if center1[0] < center2[0] else (radius2, radius1)

            midpoint = (int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2))

            # --- Define Pale Blue Color ---
            PALE_BLUE = (255, 230, 204) # BGR format

            # --- Draw New Visualizations ---
            # 1. Draw the transparent filled circle on the overlay
            cv2.circle(overlay, midpoint, 60, PALE_BLUE, -1) # 60px radius, filled

            # 2. Blend the overlay with the original frame
            alpha = 0.6 # 60% transparency
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # 3. Draw the non-transparent elements on the now-blended frame
            cv2.line(frame, left_center, right_center, PALE_BLUE, 2)
            cv2.circle(frame, left_center, left_radius, PALE_BLUE, 3)
            cv2.circle(frame, right_center, right_radius, PALE_BLUE, 3)

            # 4. Display midpoint coordinates
            cv2.putText(frame, f"FOCUS: {midpoint}", (midpoint[0] - 80, midpoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Write the processed frame to the output file
        out.write(frame)

        # Display the frame
        cv2.imshow("AI Instrument Tracker", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Application finished successfully.")

if __name__ == '__main__':
    main()

