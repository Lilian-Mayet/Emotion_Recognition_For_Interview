import os
import random
from PIL import Image, ImageDraw

# Paths
image_folder = "face/data/images"
label_folder = "face/data/labels2"

# Function to display a random image with bounding boxes
def display_random_image_with_bboxes(image_folder, label_folder):
    # Get all images
    all_images = []
    for split in ['train', 'val']:
        split_path = os.path.join(image_folder, split)
        all_images += [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(".jpg")]
    
    # Select a random image
    random_image_path = random.choice(all_images)
    random_label_path = os.path.join(label_folder, os.path.basename(random_image_path).replace('.jpg', '.txt'))
    
    # Open the image
    with Image.open(random_image_path) as img:
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Check if label file exists
        if os.path.exists(random_label_path):
            with open(random_label_path, 'r') as label_file:
                lines = label_file.readlines()
                for line in lines:
                    line = line.replace("Human face ", "")
                    x_min, y_min, x_max, y_max = map(float, line.split())
                    

                    
                    # Draw bounding box
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Display the image
        img.show()

# Run the function
display_random_image_with_bboxes(image_folder, label_folder)
