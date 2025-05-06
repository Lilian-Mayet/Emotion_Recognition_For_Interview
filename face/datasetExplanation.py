
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
# Paths
image_folder = "face/data/images"
label_folder = "face/data/labels2"


train_image_sizes = []
val_image_sizes = []
train_faces_per_image = []
val_faces_per_image = []
train_bounding_box_sizes = []
val_bounding_box_sizes = []

# Process train and val splits separately
for split in ['train', 'val']:
    image_split_path = os.path.join(image_folder, split)

    for file_name in os.listdir(image_split_path):
        if file_name.endswith(".jpg"):
            # Read corresponding label
            label_file = file_name.replace('.jpg', '.txt')
            label_path = os.path.join(label_folder, label_file)
            
            # Read image
            image_path = os.path.join(image_split_path, file_name)
            with Image.open(image_path) as img:
                width, height = img.size

            # Check if label file exists
            if os.path.exists(label_path):
                with open(label_path, 'r') as label_file:
                    lines = label_file.readlines()
                    if split == 'train':
                        train_image_sizes.append((width, height))
                        train_faces_per_image.append(len(lines))
                        for line in lines:
                            line = line.replace("Human face ", "")
                            x_min, y_min, x_max, y_max = map(float, line.split())
                            # Calculate bounding box width and height in pixels
                            bbox_width = (x_max - x_min) 
                            bbox_height = (y_max - y_min)
                            train_bounding_box_sizes.append((bbox_width, bbox_height))
                    elif split == 'val':
                        val_image_sizes.append((width, height))
                        val_faces_per_image.append(len(lines))
                        for line in lines:
                            line = line.replace("Human face ", "")
                            x_min, y_min, x_max, y_max = map(float, line.split())
                            # Calculate bounding box width and height in pixels
                            bbox_width = (x_max - x_min) 
                            bbox_height = (y_max - y_min) 
                            val_bounding_box_sizes.append((bbox_width, bbox_height))


# Prepare data for train and val plots
# Prepare data for train and val plots
train_image_widths = [size[0] for size in train_image_sizes]
train_image_heights = [size[1] for size in train_image_sizes]
train_bbox_widths = [size[0] for size in train_bounding_box_sizes]
train_bbox_heights = [size[1] for size in train_bounding_box_sizes]
train_bbox_areas = [size[0] * size[1] for size in train_bounding_box_sizes]

val_image_widths = [size[0] for size in val_image_sizes]
val_image_heights = [size[1] for size in val_image_sizes]
val_bbox_widths = [size[0] for size in val_bounding_box_sizes]
val_bbox_heights = [size[1] for size in val_bounding_box_sizes]
val_bbox_areas = [size[0] * size[1] for size in val_bounding_box_sizes]

# Plot Gaussian distribution of train and val image dimensions
plt.figure()
plt.hist(train_image_widths, bins=200, density=True, alpha=0.6, label='Train Width')
plt.hist(val_image_widths, bins=200, density=True, alpha=0.6, label='Val Width')
plt.title('Comparison of Image Widths: Train vs Val')
plt.xlabel('Pixels')
plt.ylabel('Density')
plt.legend()
plt.show()

plt.figure()
plt.hist(train_image_heights, bins=200, density=True, alpha=0.6, label='Train Height')
plt.hist(val_image_heights, bins=200, density=True, alpha=0.6, label='Val Height')
plt.title('Comparison of Image Heights: Train vs Val')
plt.xlabel('Pixels')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot distribution of number of faces per image for train and val
plt.figure()
plt.hist(train_faces_per_image, bins=range(1, max(train_faces_per_image + val_faces_per_image) + 2), 
         align='left', alpha=0.7, label='Train')
plt.hist(val_faces_per_image, bins=range(1, max(train_faces_per_image + val_faces_per_image) + 2), 
         align='left', alpha=0.7, label='Val')
plt.title('Comparison of Faces per Image: Train vs Val')
plt.xlabel('Number of Faces')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot Gaussian distribution of bounding box widths for train and val
plt.figure()
plt.hist(train_bbox_widths, bins=200, density=True, alpha=0.6, label='Train BBox Width')
plt.hist(val_bbox_widths, bins=200, density=True, alpha=0.6, label='Val BBox Width')
plt.title('Comparison of Bounding Box Widths: Train vs Val')
plt.xlabel('Width (pixels)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot Gaussian distribution of bounding box heights for train and val
plt.figure()
plt.hist(train_bbox_heights, bins=200, density=True, alpha=0.6, label='Train BBox Height')
plt.hist(val_bbox_heights, bins=200, density=True, alpha=0.6, label='Val BBox Height')
plt.title('Comparison of Bounding Box Heights: Train vs Val')
plt.xlabel('Height (pixels)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot Gaussian distribution of bounding box areas for train and val
plt.figure()
plt.hist(train_bbox_areas, bins=200, density=True, alpha=0.6, label='Train BBox Area')
plt.hist(val_bbox_areas, bins=200, density=True, alpha=0.6, label='Val BBox Area')
plt.title('Comparison of Bounding Box Areas: Train vs Val')
plt.xlabel('Area (pixels)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Summary statistics for train and val
train_summary = {
    "Split": "Train",
    "Mean Width (images)": np.mean(train_image_widths),
    "Mean Height (images)": np.mean(train_image_heights),
    "Mean Bounding Box Width": np.mean(train_bbox_widths),
    "Mean Bounding Box Height": np.mean(train_bbox_heights),
    "Mean Bounding Box Area": np.mean(train_bbox_areas),
    "Faces per Image (mean)": np.mean(train_faces_per_image),
    "Faces per Image (max)": max(train_faces_per_image),
    "Faces per Image (min)": min(train_faces_per_image),
}

val_summary = {
    "Split": "Val",
    "Mean Width (images)": np.mean(val_image_widths),
    "Mean Height (images)": np.mean(val_image_heights),
    "Mean Bounding Box Width": np.mean(val_bbox_widths),
    "Mean Bounding Box Height": np.mean(val_bbox_heights),
    "Mean Bounding Box Area": np.mean(val_bbox_areas),
    "Faces per Image (mean)": np.mean(val_faces_per_image),
    "Faces per Image (max)": max(val_faces_per_image),
    "Faces per Image (min)": min(val_faces_per_image),
}


summary_stats_df = pd.DataFrame([train_summary, val_summary])


pd.set_option('display.max_columns', None)

print("Dataset Summary Statistics:")
print(summary_stats_df)
