import streamlit as st
import requests
from PIL import Image

# Set page config
st.set_page_config(page_title="Video Interpolation", layout="centered")

# Header section
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🎬 Video Frame Interpolation App</h1>
    <p style='text-align: center;'>Upload your video and let the AI interpolate smooth frames for you!</p>
    <hr style="border:1px solid #ddd;">
""", unsafe_allow_html=True)

# Upload section
st.markdown("### 📤 Step 1: Upload Your Video")
video_file = st.file_uploader("Supported formats: MP4, MOV, AVI", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)
    st.success("✅ Video uploaded successfully!")

    # Interpolation button
    st.markdown("### ✨ Step 2: Interpolate the Video")
    if st.button("🚀 Start Interpolation"):
        with st.spinner("🔧 Processing... This may take a few seconds..."):
            files = {"video": (video_file.name, video_file, "video/mp4")}
            response = requests.post("http://localhost:8000/process_video/", files=files)

            if response.status_code == 200:
                output_path = "returned_video.mp4"
                with open(output_path, "wb") as f:
                    f.write(response.content)

                st.success("🎉 Done! Here's your interpolated video:")
                st.video(output_path)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="interpolated_video.mp4",
                        mime="video/mp4",
                        help="Click to download the processed video"
                    )
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
else:
    st.info("📎 Please upload a video file to begin.")
