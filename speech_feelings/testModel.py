import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import time
from scipy.io.wavfile import write


import os

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    return audio.flatten()

def extract_features(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)])

def predict_emotion():
    # Load the trained model
    model = tf.keras.models.load_model("speech_feelings/speech_emotion_recognition_model.h5")
    
    # Emotion labels (make sure these match the order used during training)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness']
    
    while True:
        try:
            # Record audio
            sample_rate = 16000 


            duration = 5  # seconds
            audio = record_audio(duration=duration, sample_rate=sample_rate)
            
            # Save the recorded audio (optional)
            write("temp_recording.wav", sample_rate, audio)
            
            # Extract features
            features = extract_features("temp_recording.wav")
            
            # Reshape features for model input
            features = np.expand_dims(features, axis=0)
            
            # Make prediction
            prediction = model.predict(features)
            predicted_emotion = emotions[np.argmax(prediction[0])]
            
            # Print results
            print("\nPredicted emotion:", predicted_emotion)
            print("Confidence scores:")
            for emotion, score in zip(emotions, prediction[0]):
                print(f"{emotion}: {score:.2f}")
            
            # Ask if user wants to continue
            choice = input("\nWould you like to record again? (y/n): ")
            if choice.lower() != 'y':
                break
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

if __name__ == "__main__":
    print("Speech Emotion Recognition System")
    print("--------------------------------")
    print("This program will record 5 seconds of audio and predict the emotion.")
    print("Press Enter to start recording...")
    input()
    predict_emotion()