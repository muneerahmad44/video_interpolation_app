import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_related.model_code import Model
model_path = "/home/muneer/Computer_Vision/Object_detection/tomato_detection_project_fasterrcnn/web_app_for_tomato_detection/frame_interpolation_app/backend/repo/train_log"

model = Model()
try:
    model.load_model(model_path)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
