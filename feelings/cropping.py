import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# Télécharger le modèle
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Charger le modèle
model = YOLO(model_path)

# Chemins des datasets
source_dir = "feelings/data"  # Dossier source contenant train/val
destination_dir = "feelings/data_cropped"  # Dossier destination

# Créer la structure de dossiers dans le dossier destination
for root, dirs, files in os.walk(source_dir):
    for dir_name in dirs:
        dest_path = os.path.join(destination_dir, os.path.relpath(os.path.join(root, dir_name), source_dir))
        os.makedirs(dest_path, exist_ok=True)

# Parcourir les images et les traiter
for root, _, files in os.walk(source_dir):
    for file_name in files:
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Chemin complet de l'image source
        source_path = os.path.join(root, file_name)
        # Chemin complet pour sauvegarder l'image traitée
        relative_path = os.path.relpath(source_path, source_dir)
        dest_path = os.path.join(destination_dir, relative_path)

        try:
            # Ouvrir l'image
            image = Image.open(source_path).convert("RGB")
            # Convertir en format compatible avec YOLO
            image_rgb = image.copy()
            # Effectuer la détection
            output = model(image_rgb)
            results = Detections.from_ultralytics(output[0])

            # Si aucun visage détecté, sauter l'image
            if len(results.xyxy) == 0:
                print(f"Aucun visage détecté dans {source_path}, image ignorée.")
                continue

            # Prendre la première détection (la plus probable)
            x_min, y_min, x_max, y_max = map(int, results.xyxy[0])

            # Recadrer l'image
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Redimensionner l'image
            resized_image = cropped_image.resize((96, 96))

            # Sauvegarder l'image traitée
            resized_image.save(dest_path)
            print(f"Image sauvegardée : {dest_path}")

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {source_path}: {e}")
