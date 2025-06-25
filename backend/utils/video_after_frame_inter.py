import os
import cv2
#from IPython.display import display, HTML
from base64 import b64encode

# Set the directory containing final frames
def video_after_frame_inter(output_dir: str):
    output_video_path = "final_output_video.mp4"
    
    # Get and sort frame files
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

    # Sanity check
    if not frame_files:
        print("No frames found in output directory.")
        exit()

    # Get frame dimensions from the first image
    first_frame_path = os.path.join(output_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Create VideoWriter (30 or 60 fps depending on your need)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))  # Adjust fps as needed

    # Write frames to the video
    for frame_name in frame_files:
        frame_path = os.path.join(output_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Finalize
    out.release()
    print(f"Video saved as '{output_video_path}'")

    # Optional: Display video inline
    with open(output_video_path, 'rb') as f:
        mp4 = f.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    #display(HTML(f'<video width=600 controls><source src="{data_url}" type="video/mp4"></video>'))

    return output_video_path # ✅ Return the actual binary content of the video
