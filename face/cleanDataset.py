import os

def count_faces_in_label_file(label_path):

    try:
        with open(label_path, 'r') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0

def filter_images_by_face_count(images_dir, labels_dir, max_faces=5):

    for split in ['train', 'val']:
        image_path = os.path.join(images_dir, split)
        label_path = os.path.join(labels_dir)
        
        # List all images in the directory
        images = os.listdir(image_path)
   
        for img_name in images:

            label_file = os.path.join(label_path, f"{os.path.splitext(img_name)[0]}.txt")
            

            face_count = count_faces_in_label_file(label_file)


            if face_count > max_faces:
                os.remove(os.path.join(image_path, img_name))
                os.remove(label_file)
                print(f"Removed {img_name} with {face_count} faces")

def get_largest_bounding_box(label_path):

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            largest_area = 0
            area = 0
            
            for line in lines:
                line = line.replace("Human face ", "").strip()
                x_min, y_min, x_max, y_max = map(float, line.split())
                area = (x_max - x_min) * (y_max - y_min)
                
                if area > largest_area:
                    largest_area = area
                    largest_line = line
            
            return f"Human face {largest_line}\n" if largest_line else None
    except FileNotFoundError:
        return None

def get_bounding_box(label_path):

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

            area = 0
            
            for line in lines:
                line = line.replace("Human face ", "").strip()
                x_min, y_min, x_max, y_max = map(float, line.split())
                area = (x_max - x_min) * (y_max - y_min)
                

            return area
    except FileNotFoundError:
        return None

def delete_smallBB(images_dir, labels_dir, min_size=6000):
   
    for split in ['train', 'val']:
        image_path = os.path.join(images_dir, split)
        label_path = os.path.join(labels_dir)
        

        images = os.listdir(image_path)
        
        for img_name in images:

            label_file = os.path.join(label_path, f"{os.path.splitext(img_name)[0]}.txt")
            
            # Get the largest bounding box
            area = get_bounding_box(label_file)
            
            if area<min_size:
                os.remove(os.path.join(image_path, img_name))
                os.remove(label_file)
                print(f"Removed {img_name} for too small boundings size")

def keep1BB(images_dir, labels_dir, max_faces=5):
   
    for split in ['train', 'val']:
        image_path = os.path.join(images_dir, split)
        label_path = os.path.join(labels_dir)
        

        images = os.listdir(image_path)
        
        for img_name in images:

            label_file = os.path.join(label_path, f"{os.path.splitext(img_name)[0]}.txt")
            

            largest_bbox_line = get_largest_bounding_box(label_file)
            
            if largest_bbox_line:
                # Overwrite the label file with only the largest bounding box
                with open(label_file, 'w') as f:
                    f.write(largest_bbox_line)
            else:
                # If no valid bounding box, remove the image and label
                os.remove(os.path.join(image_path, img_name))
                os.remove(label_file)
                print(f"Removed {img_name} as no valid bounding box found")




keep1BB("face/data/images","face/data/labels2")
delete_smallBB("face/data/images","face/data/labels2",6000)