import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

# Télécharger le modèle
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Charger le modèle
model = YOLO(model_path)

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break

    # Convertir l'image en format compatible avec YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Effectuer l'inférence
    output = model(frame_rgb)
    results = Detections.from_ultralytics(output[0])

    # Dessiner les bounding boxes
    for box, score in zip(results.xyxy, results.confidence):
        x_min, y_min, x_max, y_max = map(int, box)  # Convertir les coordonnées en entiers
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, f"{score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Afficher l'image avec les rectangles
    cv2.imshow("Webcam - Détection de visage", frame)

    # Quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
