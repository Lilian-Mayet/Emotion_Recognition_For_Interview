import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense, Dropout, BatchNormalization
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint  
from tensorflow.keras.saving import register_keras_serializable

# Config
IMG_SIZE = (128,128)
BATCH_SIZE = 8
EPOCHS = 10
DATA_DIR = "face/data" 
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

# Data augmentation
augmentation = Compose([
    Resize(IMG_SIZE[0], IMG_SIZE[1]),
], bbox_params={'format': 'pascal_voc', 'label_fields': []})


def preprocess_image(img_path, label_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    original_height = tf.shape(img)[0]
    original_width = tf.shape(img)[1]

    img = tf.image.resize(img, IMG_SIZE) / 255.0  


    label = tf.io.read_file(label_path)
    label = tf.strings.regex_replace(label, "Human face ", "")
    label = tf.strings.to_number(tf.strings.split(label), tf.float32)

    # Conversion des coordonnées en fonction des dimensions redimensionnées
    label = tf.stack([
        label[0] / tf.cast(original_width, tf.float32) * IMG_SIZE[0],  # x_min
        label[1] / tf.cast(original_height, tf.float32) * IMG_SIZE[1], # y_min
        label[2] / tf.cast(original_width, tf.float32) * IMG_SIZE[0],  # x_max
        label[3] / tf.cast(original_height, tf.float32) * IMG_SIZE[1]  # y_max
    ])

    # Normalisation entre 0 et 1 (par rapport aux dimensions redimensionnées)
    label = tf.stack([
        label[0] / IMG_SIZE[0],  # x_min normalisé
        label[1] / IMG_SIZE[1], # y_min normalisé
        label[2] / IMG_SIZE[0], # x_max normalisé
        label[3] / IMG_SIZE[1]  # y_max normalisé
    ])
    return img, label


def load_dataset(image_dir, label_dir, batch_size=BATCH_SIZE):

    img_paths = tf.data.Dataset.list_files(os.path.join(image_dir, "*.jpg"))
    label_paths = tf.data.Dataset.list_files(os.path.join(label_dir, "*.txt"))

    # Pair image paths with corresponding label paths
    dataset = tf.data.Dataset.zip((img_paths, label_paths))

    # Preprocess images and labels
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset




train_dataset = load_dataset(os.path.join(DATA_DIR, "images/train"), 
                             os.path.join(DATA_DIR, "labels2"), 
                             batch_size=BATCH_SIZE)

val_dataset = load_dataset(os.path.join(DATA_DIR, "images/val"), 
                           os.path.join(DATA_DIR, "labels2"), 
                           batch_size=BATCH_SIZE)

# Model creation
# Build the model
def build_model():
    input_layer = Input(shape=(IMG_SIZE[0],IMG_SIZE[1], 3))  # Taille de l'entrée : 224x224x3
    

    vgg = VGG16(include_top=False, weights='imagenet')(input_layer)


    x = GlobalMaxPooling2D()(vgg)


    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)  # Normalisation
    x = Dropout(0.3)(x)  # Trop de suraprentissage
    
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)


    output = Dense(4, activation='sigmoid')(x)  # Activation sigmoïde pour des valeurs entre 0 et 1

    # Modèle final
    facetracker = Model(inputs=input_layer, outputs=output)
    return facetracker


# Custom FaceTracker model for training
@register_keras_serializable()
class FaceTracker(Model): 
    def __init__(self, facetracker, **kwargs): 
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        X, y = batch
        
        with tf.GradientTape() as tape: 
            coords = self.model(X, training=True)
            batch_localizationloss = self.lloss(tf.cast(y, tf.float32), coords)
            total_loss = batch_localizationloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss": total_loss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        coords = self.model(X, training=False)
        batch_localizationloss = self.lloss(tf.cast(y, tf.float32), coords)
        total_loss = batch_localizationloss
        
        return {"total_loss": total_loss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

facetracker = build_model()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
regressloss = tf.keras.losses.MeanSquaredError()

model = FaceTracker(facetracker)
model.compile(opt=opt, localizationloss=regressloss)


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Dir pour sauvegarder les checkpoints
checkpoint_dir = "face/model/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Callback pour sauvegarder les checkpoints après chaque epoch
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "facetracker_epoch_{epoch:02d}.keras"),  # Fichier avec numéro d'épocqh
    save_weights_only=False,  # Sauvegarder tout le modèle 
    save_best_only=False,     # Sauvegarder à chaque époque 
    monitor='val_total_loss', 
    mode='min',               
    verbose=1                 # Affiche les informations de sauvegarde
)
STEPS_PER_EPOCH = 128  # Nombre de batches/epcoh
hist = model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=STEPS_PER_EPOCH,  # Limiter les batches par époch
    validation_data=val_dataset,
    validation_steps=50,
    callbacks=[tensorboard_callback,checkpoint_callback]
)


final_model_path = os.path.join("face/model", "final_facetracker_model.keras")
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)  
model.save(final_model_path)  

print(f"Modèle final sauvegardé à {final_model_path}")

# Plot Perf
fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

plt.show()
