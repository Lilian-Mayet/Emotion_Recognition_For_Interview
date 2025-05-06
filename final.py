import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(model_path)

emotion_model = tf.keras.models.load_model("feelings/best_model.keras")
emotion_classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Fonction pour redimensionner une image tout en maintenant le ratio d'agrandissement
def crop_and_resize(frame, bbox, scale=1.25, target_size=(96, 96)):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = x_min + w // 2, y_min + h // 2

    # Agrandir la bounding box
    new_w, new_h = int(w * scale), int(h * scale)
    x_min = int(max(cx - new_w // 2, 0))
    y_min = int(max(cy - new_h // 2, 0))
    x_max = int(min(cx + new_w // 2, frame.shape[1]))
    y_max = int(min(cy + new_h // 2, frame.shape[0]))

    # Recadrer et redimensionner l'image
    cropped_face = frame[y_min:y_max, x_min:x_max]
    resized_face = cv2.resize(cropped_face, target_size)
    return resized_face, (x_min, y_min, x_max, y_max)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break

    # Détection de visages
    results = face_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # Coordonnées 
    confidences = results[0].boxes.conf.cpu().numpy()  

    for bbox, conf in zip(detections, confidences):
        if conf < 0.5:  # Filtre
            continue

        # Recadrer et redimensionner le visage détecté
        face, (x_min, y_min, x_max, y_max) = crop_and_resize(frame, bbox)

        #pour le modèle d'émotion
        face_input = np.expand_dims(face / 255.0, axis=0)

        # Prédire l'expression faciale
        predictions = emotion_model.predict(face_input, verbose=0)[0]
        #retouche manuelle des probabilités
        predictions[5]*=0.92#harder to get neutral
        predictions[4]*=1.1#easier to get happy
        predictions[7]*=0.91#harder to get surprise
        emotion_idx = np.argmax(predictions)
        emotion_label = emotion_classes[emotion_idx]
        confidence = predictions[emotion_idx]

        # Dessiner la bounding box et afficher l'émotion prédite
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{emotion_label}: {confidence:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher la vidéo
    cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Webcam", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
