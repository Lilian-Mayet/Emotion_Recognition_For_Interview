import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib.patches import Rectangle


# Dimensions des images
IMG_SIZE = (200, 200)

# Répertoires des données
IMAGES_DIR = "face/data/images/val"
LABELS_DIR = "face/data/labels2"

# Charger le modèle entraîné
MODEL_PATH = "face/model/checkpoints/epoch_07.keras"  # Changez le chemin si nécessaire
MODEL_PATH = "face/face_detection_model.keras"
if not os.path.exists(MODEL_PATH):
    print(f"Le fichier modèle n'existe pas à l'emplacement : {MODEL_PATH}")
model = load_model(MODEL_PATH,safe_mode = False )

def load_random_image(images_dir, labels_dir):
    """Charge une image aléatoire et ses étiquettes."""
    image_name = random.choice(os.listdir(images_dir))
    img_path = os.path.join(images_dir, image_name)
    label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    
    # Charger l'image originale
    original_img = load_img(img_path)
    original_width, original_height = original_img.size
    
    # Charger et redimensionner l'image pour le modèle
    img = original_img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalisation entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    
    # Charger les étiquettes correspondantes
    true_boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("Human face ", "")
            x_min, y_min, x_max, y_max = map(float, line.split())
            # Adapter les coordonnées à l'image redimensionnée
            x_min = (x_min / original_width) * IMG_SIZE[0]
            y_min = (y_min / original_height) * IMG_SIZE[1]
            x_max = (x_max / original_width) * IMG_SIZE[0]
            y_max = (y_max / original_height) * IMG_SIZE[1]
            true_boxes.append((x_min, y_min, x_max, y_max))
    
    return original_img, img_array, true_boxes


def rescale_boxes(boxes, original_width, original_height, resized_width, resized_height):
    """Recalcule les coordonnées des boîtes pour qu'elles correspondent à la taille réelle de l'image."""
    scaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = (x_min / resized_width) * original_width
        y_min = (y_min / resized_height) * original_height
        x_max = (x_max / resized_width) * original_width
        y_max = (y_max / resized_height) * original_height
        scaled_boxes.append((x_min, y_min, x_max, y_max))
    return scaled_boxes


def plot_image_with_boxes(original_img, true_boxes, predicted_boxes):
    """Affiche une image avec les boîtes réelles et prédites."""
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    ax = plt.gca()

    # Tracer les vraies boîtes (en vert)
    for box in true_boxes:
        x_min, y_min, x_max, y_max = box
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    # Tracer les boîtes prédites (en bleu)
    for box in predicted_boxes:
        x_min, y_min, x_max, y_max = box
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    legend_handles = [
        Rectangle((0, 0), 1, 1, edgecolor='green', facecolor='none', label='True Box'),
        Rectangle((0, 0), 1, 1, edgecolor='blue', facecolor='none', label='Predicted Box')
    ]
    plt.legend(handles=legend_handles, loc='upper left')
    plt.show()

while True:
    # Charger une image aléatoire et ses étiquettes
    original_img, img_array, true_boxes = load_random_image(IMAGES_DIR, LABELS_DIR)

    # Faire des prédictions
    predictions = model.predict(img_array)  # Shape: (1, MAX_FACES, 4)
    predictions = predictions.reshape(-1, 4)  # Remove batch dimension for easier handling
    print(predictions)
    # Adapter les coordonnées des prédictions à l'image redimensionnée
    predicted_boxes = rescale_boxes(predictions, original_img.size[0], original_img.size[1], IMG_SIZE[0], IMG_SIZE[1])
    true_boxes = rescale_boxes(true_boxes, original_img.size[0], original_img.size[1], IMG_SIZE[0], IMG_SIZE[1])
    
    # Afficher l'image avec les boîtes
    plot_image_with_boxes(original_img, true_boxes, predicted_boxes)
