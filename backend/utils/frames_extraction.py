import cv2
import os

def extract_frames(video_path):
    # Create absolute path for 'input_frames' folder in the same directory as the video
    base_dir = os.path.dirname(os.path.abspath(video_path))
    frames_dir = os.path.join(base_dir, "input_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Read video
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    if not success:
        print(f"Failed to read video: {video_path}")
        return None

    count = 0
    while success:
        filename = os.path.join(frames_dir, f"frame_{count:04d}.png")
        cv2.imwrite(filename, image)
        success, image = cap.read()
        count += 1

    cap.release()
    print(f"Extracted {count} frames to '{frames_dir}'")
    return frames_dir  # ✅ Return path to 'input_frames'

# Example usage:
# path = extract_frames("/path/to/video.mp4")
# print("Frames saved at:", path)
