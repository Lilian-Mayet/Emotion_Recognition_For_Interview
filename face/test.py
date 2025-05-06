# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import random
from PIL import Image
from PIL import Image, ImageDraw
import os

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
while True:
    VAL_IMAGE_DIR = "face/data/images/val"  # Dossier contenant les images de validation
    img_name = random.choice(os.listdir(VAL_IMAGE_DIR))

    image_path = os.path.join(VAL_IMAGE_DIR, img_name)
    image = Image.open(image_path)

    output = model(image)
    print(output)
    results = Detections.from_ultralytics(output[0])
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box, score in zip(results.xyxy, results.confidence):
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min), f"{score:.2f}", fill="red")

    # Show the image with bounding boxes
    image.show()

