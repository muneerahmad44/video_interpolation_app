import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from backend.utils.frames_extraction import extract_frames
from backend.utils.frames_interpolation import frames_interpolation
from backend.utils.video_after_frame_inter import video_after_frame_inter

app = FastAPI()

@app.post("/process_video/")
async def process_video(video: UploadFile = File(...)):
    # Save uploaded video to temp folder
    os.makedirs("temp_videos", exist_ok=True)
    temp_video_path = os.path.join("temp_videos", video.filename)

    with open(temp_video_path, "wb") as f:
        f.write(await video.read())

    # Process video
    input_frames_path = extract_frames(temp_video_path)
    interpolated_frames_path = frames_interpolation(input_frames_path)
    #= "final_output_video.mp4"  # This is where the final video is saved
    output_video_path =video_after_frame_inter(interpolated_frames_path)  # Saves the video to that path

    # Return video as StreamingResponse
    return StreamingResponse(
        open(output_video_path, "rb"),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={os.path.basename(output_video_path)}"}
    )
