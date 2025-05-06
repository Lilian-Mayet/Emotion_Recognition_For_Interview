import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration des paramètres
IMG_SIZE = (96, 96)  # Taille des images
BATCH_SIZE = 32  # Taille du batch
EPOCHS = 50  # Nombre d'époques
LEARNING_RATE = 0.001  # Taux d'apprentissage
MODEL_PATH = "best_model.keras"  # Chemin pour enregistrer le meilleur modèle
CLASSES = ["anger","contempt","disgust","fear","happy","neutral","sad","surprise"]  # Classes disponibles (peut être modifié selon votre dataset)

# Répertoires du dataset
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# Préparation des générateurs de données
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalisation des pixels entre 0 et 1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES
)

# Création du modèle
def create_model(input_shape=(96, 96, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Initialisation du modèle
model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(CLASSES))

# Compilation du modèle
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
]

# Entraînement du modèle
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# Évaluation finale
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Sauvegarder le modèle final
model.save("final_model.h5")
