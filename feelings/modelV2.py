import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt  # Ajout pour la visualisation
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LearningRateScheduler
MODEL_PATH = "best_model.keras"  # Chemin vers le modèle sauvegardé


# Configuration des paramètres globaux
def configure_settings():
    return {
        "IMG_SIZE": (96, 96),
        "BATCH_SIZE": 32,
        "EPOCHS": 90,
        "LEARNING_RATE": 0.0005,
        "MODEL_PATH": "best_model.keras",
        "CLASSES": ["anger","contempt","disgust","fear","happy","neutral","sad","surprise"],  # Modifier selon les classes de votre dataset
        "TRAIN_DIR": "feelings/data/train",
        "VAL_DIR": "feelings/data/val"
    }

def lr_decay_schedule(epoch, lr):
    decay_rate = 0.95  # Decay rate
    return lr * decay_rate
# Préparation des générateurs de données
def create_data_generators(config):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
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
        config["TRAIN_DIR"],
        target_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        class_mode="categorical",
        classes=config["CLASSES"]
    )

    val_generator = val_datagen.flow_from_directory(
        config["VAL_DIR"],
        target_size=config["IMG_SIZE"],
        batch_size=config["BATCH_SIZE"],
        class_mode="categorical",
        classes=config["CLASSES"]
    )

    return train_generator, val_generator


# Création du modèle CNN
def create_model(input_shape=(96, 96, 3), num_classes=8):
    model = models.Sequential()

    # Bloc de convolution 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bloc de convolution 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bloc de convolution 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten et couches denses
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Compilation du modèle
def compile_model(model, learning_rate):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


# Création des callbacks avec apprentissage adaptatif
def create_callbacks(model_path):

    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),

    ]



# Entraînement du modèle
def train_model(model, train_generator, val_generator, epochs, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )
    return history


# Évaluation finale
def evaluate_model(model, val_generator):
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Tracer l'évolution de la précision
    plt.figure(figsize=(8, 6))
    plt.plot(acc, label='Accuracy (train)')
    plt.plot(val_acc, label='Accuracy (val)')
    plt.title('Evolution of Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Tracer l'évolution de la perte
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='Loss (train)')
    plt.plot(val_loss, label='Loss (val)')
    plt.title('Evolution of Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Programme principal
def main():
    # Charger les paramètres
    config = configure_settings()

    # Préparer les générateurs de données
    train_generator, val_generator = create_data_generators(config)

    # Créer le modèle
    model = create_model(input_shape=(config["IMG_SIZE"][0], config["IMG_SIZE"][1], 3),
                         num_classes=len(config["CLASSES"]))
    #model = load_model(MODEL_PATH)
    # Compiler le modèle
    compile_model(model, learning_rate=config["LEARNING_RATE"])

    # Créer les callbacks
    callbacks = create_callbacks(config["MODEL_PATH"])

    # Entraîner le modèle
    print("Starting training...")
    history = train_model(model, train_generator, val_generator, config["EPOCHS"], callbacks)

    # Évaluer le modèle
    print("Evaluating model...")
    evaluate_model(model, val_generator)

    # Sauvegarder le modèle final
    model.save("feelings/final_model.keras")
    
    plot_training_history(history)

if __name__ == "__main__":
    main()
