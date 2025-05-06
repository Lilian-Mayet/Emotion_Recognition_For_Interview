import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import sounddevice as sd
import librosa
import time
from scipy.io.wavfile import write
import threading
from collections import deque

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(model_path)


emotion_model = tf.keras.models.load_model("feelings/best_model.keras")
emotion_classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


audio_emotion_model = tf.keras.models.load_model("speech_feelings\speech_emotion_recognition_model.h5")
audio_emotion_classes = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


ROLLING_AVERAGE_WINDOW = 10

face_buffers = {}

# Fonction pourl'audio
def record_audio(duration=2.5, sample_rate=16000):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  
    return audio.flatten()

# Fonction pour extraire les features audio
def extract_features(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)])

# Fonction pour analyser les émotions audio
def analyze_audio_emotion(audio_path):

    features = extract_features(audio_path)
    audio_input = np.expand_dims(features, axis=0)


    #retouche manuelle des probabilités
    predictions = audio_emotion_model.predict(audio_input, verbose=0)[0]
    predictions[2]*=0.85#harder to get fear
    predictions[4]*=0.9#harder to get neutral
   
    emotion_idx = np.argmax(predictions)
    return audio_emotion_classes[emotion_idx], predictions[emotion_idx]

# Fonction pour redimensionner une image tout en maintenant le ratio d'agrandissement
def crop_and_resize(frame, bbox, scale=1.25, target_size=(96, 96)):
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    cx, cy = x_min + w // 2, y_min + h // 2

    new_w, new_h = int(w * scale), int(h * scale)
    x_min = int(max(cx - new_w // 2, 0))
    y_min = int(max(cy - new_h // 2, 0))
    x_max = int(min(cx + new_w // 2, frame.shape[1]))
    y_max = int(min(cy + new_w // 2, frame.shape[0]))


    cropped_face = frame[y_min:y_max, x_min:x_max]
    resized_face = cv2.resize(cropped_face, target_size)
    return resized_face, (x_min, y_min, x_max, y_max)


# Fonction pour cacluler les predictions sur une fenetre roulante
def update_rolling_average(face_id, predictions, buffer_dict, window_size):
    if face_id not in buffer_dict:
        buffer_dict[face_id] = deque(maxlen=window_size)
    buffer_dict[face_id].append(predictions)
    # Compute the average prediction
    rolling_average = np.mean(buffer_dict[face_id], axis=0)
    return rolling_average
# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Appuyez sur 'q' pour quitter.")

audio_buffer = []
audio_sample_rate = 16000
last_audio_time = time.time()

audio_sentiment = "Neutral"
audio_confidence = 0.0

def process_audio():
    global audio_sentiment, audio_confidence
    while True:
        # Record audio
        audio_data = record_audio(duration=2.5, sample_rate=audio_sample_rate)
        audio_path = "temp_audio.wav"
        write(audio_path, audio_sample_rate, audio_data)

        try:
            sentiment, confidence = analyze_audio_emotion(audio_path)

        except Exception as e:
            print(f"Erreur lors de l'analyse audio : {e}")
            sentiment, confidence = "Error", 0.0

        
        with audio_lock:
            audio_sentiment = sentiment
            audio_confidence = confidence
audio_lock = threading.Lock()

audio_thread = threading.Thread(target=process_audio, daemon=True)
audio_thread.start()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break


    results = face_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy() 
    confidences = results[0].boxes.conf.cpu().numpy()  

    for idx, (bbox, conf) in enumerate(zip(detections, confidences)):
        if conf < 0.5:
            continue

        face, (x_min, y_min, x_max, y_max) = crop_and_resize(frame, bbox)
        face_input = np.expand_dims(face / 255.0, axis=0)
        predictions = emotion_model.predict(face_input, verbose=0)[0]
        #retouche manuelle des probabilités
        predictions[5]*=0.85#harder to get neutral
        predictions[4]*=1.1#easier to get happy
        predictions[3]*=1.2#easier to get fear
        predictions[7]*=0.9#harder to get surprise
        predictions[6]*=1.4#easier to get sad
        rolling_predictions = update_rolling_average(idx, predictions, face_buffers, ROLLING_AVERAGE_WINDOW)
        emotion_idx = np.argmax(rolling_predictions)
        emotion_label = emotion_classes[emotion_idx]
        if emotion_label=="surprise":
            emotion_label = "happy"
        confidence = predictions[emotion_idx]

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{emotion_label}: {confidence:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Affichage
    with audio_lock:
        audio_label = f"Audio Sentiment: {audio_sentiment} ({audio_confidence:.2f})"
    cv2.putText(frame, audio_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    
 
    cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
