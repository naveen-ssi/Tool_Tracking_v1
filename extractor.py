import cv2
import os

# --- Configuration ---
VIDEO_SOURCE = "BIMA MANTRA.mp4" # The video you want to extract frames from
OUTPUT_FOLDER = "new_images_2000"   # The folder where the extracted images will be saved
DESIRED_IMAGE_COUNT = 2000     # The approximate number of images you want to extract
# --------------------

def main():
    """
    Extracts a specific number of frames from a video file and saves them
    as individual image files.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created directory: {OUTPUT_FOLDER}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'")
        return

    # --- Calculate the correct frame interval ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print("Error: Could not get total frame count from video. Is the file path correct?")
        return
        
    # Calculate the interval to get the desired number of images
    # Use max(1, ...) to avoid an interval of 0
    frame_interval = max(1, int(total_frames / DESIRED_IMAGE_COUNT))
    print(f"Video has {total_frames} total frames.")
    print(f"To get approximately {DESIRED_IMAGE_COUNT} images, we will save one frame every {frame_interval} frames.")
    # ---------------------------------------------

    frame_count = 0
    saved_count = 0

    print(f"\nStarting frame extraction from '{VIDEO_SOURCE}'...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished extracting all frames.")
            break

        # Check if it's time to save a frame based on the calculated interval
        if frame_count % frame_interval == 0:
            # Construct the output filename
            image_filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved_count:04d}.jpg")
            
            # Save the frame as a JPEG image
            cv2.imwrite(image_filename, frame)
            
            # Print progress every 100 saves to avoid flooding the terminal
            if saved_count > 0 and saved_count % 100 == 0:
                print(f"Saved {saved_count} images...")

            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"\n✅ Success! Extracted a total of {saved_count} frames.")

if __name__ == '__main__':
    main()

