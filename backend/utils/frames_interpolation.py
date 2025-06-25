import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import torch.nn.functional as F
from model_related.load_model import model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def frames_interpolation(frames_dir: str):
    # Path to the frames folder
    
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    # Transformation: convert PIL images to tensor normalized between 0 and 1
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    frame_files = sorted(os.listdir(frames_dir))
    frame_files = [f for f in frame_files if f.endswith('.png')]

    count = 0
    for i in range(len(frame_files) - 1):
        # Load two consecutive frames
        img0_path = os.path.join(frames_dir, frame_files[i])
        img1_path = os.path.join(frames_dir, frame_files[i + 1])

        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")

        img0_t = transform(img0).unsqueeze(0).to(device)
        img1_t = transform(img1).unsqueeze(0).to(device)

        img0_t = F.interpolate(img0_t, size=(480, 688), mode='bilinear', align_corners=False)
        img1_t = F.interpolate(img1_t, size=(480, 688), mode='bilinear', align_corners=False)

        print(img0_t.shape)
        # Run interpolation at t=0.5
        with torch.no_grad():
            middle_frame = model.inference(img0_t, img1_t, timestep=0.5)

        # middle_frame is tensor in [0,1], convert to PIL image
        middle_img = transforms.ToPILImage()(middle_frame.squeeze(0).cpu().clamp(0, 1))

        # Save img0, interpolated frame, then img1 (for the first iteration save img0, for others skip to avoid duplicates)
        if i == 0:
            img0.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
            count += 1

        middle_img.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
        count += 1

        img1.save(os.path.join(output_dir, f"frame_{count:04d}.png"))
        count += 1

    print(f"Saved {count} frames with interpolation to '{output_dir}'.")
    return output_dir  # ✅ Only this line is added
