import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Preparation
def extract_features(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    return np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)])

# Example directory and labels
data_directory = "speech_feelings/data"
labels = []
features = []
weights = []  # New array to store intensity weights

emotion_mapping = {
    "SAD": "sadness",
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral"
}

intensity_mapping = {
    "HI": 1.0,    # High intensity gets full weight
    "MD": 0.7,    # Medium intensity gets 0.7 weight
    "LO": 0.4,    # Low intensity gets 0.4 weight
    "XX": 0.7   # XX will be filtered out
}

for subdir, dirs, files in os.walk(data_directory):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            file_parts = file.split("_")
            emotion_code = file_parts[2]
            intensity_code = file_parts[3] if len(file_parts) > 3 else "XX"
            intensity_code=  intensity_code.replace(".wav","")

            
            emotion_label = emotion_mapping.get(emotion_code, "unknown")
            intensity_weight = intensity_mapping.get(intensity_code)
            
            if emotion_label != "unknown" and intensity_weight is not None:
                features.append(extract_features(file_path))
                labels.append(emotion_label)
                weights.append(intensity_weight)
                print(f"{file} - {emotion_label} - Intensity: {intensity_code} (weight: {intensity_weight})")

# Encoding labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Converting to NumPy arrays
features = np.array(features)
labels_encoded = np.array(labels_encoded)
weights = np.array(weights)

# Train-test split with weights
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    features, labels_encoded, weights, test_size=0.2, random_state=42
)

# Step 2: Model Creation
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_training_history(history):
    """
    Trace les courbes d'évolution de l'accuracy et de la loss pendant l'entraînement.
    
    Args:
        history: L'objet History retourné par model.fit()
    """
    # Récupérer les données de l'historique
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Tracer l'évolution de l'accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(acc, label='Accuracy (train)', marker='o')
    plt.plot(val_acc, label='Accuracy (val)', marker='o')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tracer l'évolution de la loss
    plt.figure(figsize=(12, 6))
    plt.plot(loss, label='Loss (train)', marker='o')
    plt.plot(val_loss, label='Loss (val)', marker='o')
    plt.title("Évolution de la Loss")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


input_shape = (X_train.shape[1],)
print(input_shape)
num_classes = len(np.unique(y_train))
model = create_model(input_shape, num_classes)

# Step 3: Training with sample weights
#lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(-epoch / 20))

early_stopping = EarlyStopping(
    monitor='val_loss',  # Surveiller la perte de validation
    patience=50,         # Arrêter après 10 époques sans amélioration
    restore_best_weights=True,  # Rétablir les meilleurs poids trouvés
    verbose=1            # Afficher les logs
)
history = model.fit(
    X_train, 
    y_train, 
    epochs=500, 
    batch_size=32, 
    validation_split=0.2, 
    sample_weight=weights_train,  # Add sample weights to training
    verbose=1,
    callbacks=[early_stopping]
)
plot_training_history(history)

# Step 4: Save the Model
model.save("speech_feelings/speech_emotion_recognition_model.h5")

def test_model(test_features, test_labels, test_weights):
    # Load the saved model
    loaded_model = tf.keras.models.load_model("speech_feelings/speech_emotion_recognition_model.h5")
    
    # Get predictions
    predictions = loaded_model.predict(test_features)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate weighted accuracy
    correct_predictions = (predicted_classes == test_labels)
    weighted_accuracy = np.sum(correct_predictions * test_weights) / np.sum(test_weights)
    print(f"Weighted Test Accuracy: {weighted_accuracy:.2f}")
    
    # Calculate standard accuracy for comparison
    standard_accuracy = np.mean(correct_predictions)
    print(f"Standard Test Accuracy: {standard_accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    # Map integer labels to emotion names
    class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Evaluate the loaded model
test_model(X_test, y_test, weights_test)