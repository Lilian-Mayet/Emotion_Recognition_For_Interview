import os
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Confi
IMG_SIZE = (128, 128)  # Taille d'entrée pour le modèle

MODEL_PATH = "face/model/checkpoints/facetracker_epoch_01.keras"
VAL_IMAGE_DIR = "face/data/images/val"  


model = tf.keras.models.load_model(MODEL_PATH)


def load_random_image(image_dir):
    img_name = random.choice(os.listdir(image_dir))
    img_path = os.path.join(image_dir, img_name)

    # Charger l'image originale
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_height, original_width = original_img.shape[:2]

    # Redimensionner et normaliser l'image pour le modèle
    resized_img = cv2.resize(original_img, IMG_SIZE)
    input_img = resized_img / 255.0  # Normalisation entre 0 et 1
    input_img = np.expand_dims(input_img, axis=0)  # Ajouter une dimension pour le batch

    return img_name, original_img, input_img, original_width, original_height

# Fonction pour effectuer une prédiction et redimensionner les coordonnées
def predict_and_resize(img, original_width, original_height):
    # Prédire les coordonnées de la boîte englobante (normalisées entre 0 et 1)
    pred_bbox = model.predict(img)[0]

    # Convertir les coordonnées en pixels (dans l'espace 224x224)
    pred_bbox = [
        pred_bbox[0] * IMG_SIZE[0],  # x_min
        pred_bbox[1] * IMG_SIZE[1],  # y_min
        pred_bbox[2] * IMG_SIZE[0],  # x_max
        pred_bbox[3] * IMG_SIZE[1]   # y_max
    ]

    # Redimensionner les coordonnées pour correspondre à l'image originale
    pred_bbox = [
        pred_bbox[0] * (original_width / IMG_SIZE[0]),  # x_min original
        pred_bbox[1] * (original_height / IMG_SIZE[1]), # y_min original
        pred_bbox[2] * (original_width / IMG_SIZE[0]),  # x_max original
        pred_bbox[3] * (original_height / IMG_SIZE[1])  # y_max original
    ]

    return pred_bbox

# Fonction pour tracer l'image avec la boîte englobante
def plot_image_with_bbox(img, bbox, title="Image avec boîte englobante"):
    # Extraire les coordonnées
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Dessiner la boîte englobante sur l'image
    img_with_bbox = img.copy()
    cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Afficher l'image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_bbox)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Tester le modèle
if __name__ == "__main__":
    # Charger une image aléatoire
    img_name, original_img, input_img, original_width, original_height = load_random_image(VAL_IMAGE_DIR)

    # Prédire les coordonnées de la boîte englobante
    pred_bbox = predict_and_resize(input_img, original_width, original_height)

    # Afficher le résultat
    print(f"Prédiction pour {img_name}: {pred_bbox}")
    plot_image_with_bbox(original_img, pred_bbox)
