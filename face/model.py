import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, Affine
from albumentations.core.bbox_utils import BboxParams
from tensorflow.keras.saving import register_keras_serializable
import cv2


IMAGES_DIR = "face/data/images"
LABELS_DIR = "face/data/labels2"

# Image dimensions
IMG_SIZE = (200, 200)
BATCH_SIZE = 60

def normalize_bbox(bbox, width, height):

    x_min, y_min, x_max, y_max = bbox
    return [
        x_min / width,
        y_min / height,
        x_max / width,
        y_max / height
    ]

def denormalize_bbox(bbox, width, height):

    x_min, y_min, x_max, y_max = bbox
    return [
        x_min * width,
        y_min * height,
        x_max * width,
        y_max * height
    ]

def data_generator(images_dir, labels_dir, batch_size=1):
 
    image_names = os.listdir(images_dir)
    total_images = len(image_names)

    augmentation = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=60, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    while True:
        np.random.shuffle(image_names)

        for start in range(0, total_images, batch_size):
            batch_images = []
            batch_labels = []

            for image_name in image_names[start:start + batch_size]:
                img_path = os.path.join(images_dir, image_name)
                img = load_img(img_path)
                original_width, original_height = img.size
                img = img.resize(IMG_SIZE)
                img = img_to_array(img) / 255.0

                label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().replace("Human face ", "").strip()
                        x_min, y_min, x_max, y_max = map(float, line.split())

                        bounding_box = [
                            (x_min / original_width) * IMG_SIZE[0],
                            (y_min / original_height) * IMG_SIZE[1],
                            (x_max / original_width) * IMG_SIZE[0],
                            (y_max / original_height) * IMG_SIZE[1]
                        ]

                        augmented = augmentation(image=img, bboxes=[bounding_box])

                        if augmented['bboxes']:
                            batch_labels.append(augmented['bboxes'][0])
                            batch_images.append(augmented['image'])
                        else:
                            batch_labels.append(bounding_box)
                            batch_images.append(img)
                except FileNotFoundError:
                    print(f"Label file not found for {image_name}")
                    continue

            yield np.array(batch_images), np.array(batch_labels)

def create_improved_face_model(input_shape):

    inputs = layers.Input(shape=input_shape)
    
    # Encoder path with skip connections
    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    block1_out = layers.Activation('relu')(x)
    
    # Attention 1
    attention1 = layers.Conv2D(64, (1, 1), activation='sigmoid')(block1_out)
    block1_out = layers.Multiply()([block1_out, attention1])
    x = layers.MaxPooling2D((2, 2))(block1_out)
    
    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    block2_out = layers.Activation('relu')(x)
    
    # Attention 2
    attention2 = layers.Conv2D(128, (1, 1), activation='sigmoid')(block2_out)
    block2_out = layers.Multiply()([block2_out, attention2])
    x = layers.MaxPooling2D((2, 2))(block2_out)
    
    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global features
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with strong regularization
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output with sigmoid activation for normalized coordinates
    outputs = layers.Dense(4)(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def visualize(image, bboxes, normalized=True):

    import matplotlib.pyplot as plt
    import cv2
    
    img_height, img_width = image.shape[:2]
    image_display = (image * 255).astype(np.uint8).copy()
    
    for bbox in bboxes:
        if normalized:
            bbox = denormalize_bbox(bbox, img_width, img_height)
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image_display, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    plt.imshow(image_display)
    plt.show()

@register_keras_serializable()
def iou_loss(y_true, y_pred):

    # Coordonnées des boîtes réelles et prédites
    x_min_true, y_min_true, x_max_true, y_max_true = tf.split(y_true, 4, axis=-1)
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = tf.split(y_pred, 4, axis=-1)

    # Calcul des coordonnées de l'intersection
    inter_x_min = tf.maximum(x_min_true, x_min_pred)
    inter_y_min = tf.maximum(y_min_true, y_min_pred)
    inter_x_max = tf.minimum(x_max_true, x_max_pred)
    inter_y_max = tf.minimum(y_max_true, y_max_pred)

    # Aire de l'intersection
    inter_area = tf.maximum(0.0, inter_x_max - inter_x_min) * tf.maximum(0.0, inter_y_max - inter_y_min)

    # Aire des boîtes réelles et prédites
    true_area = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)

    # Aire de l'union
    union_area = true_area + pred_area - inter_area
    print(union_area)

    # Calcul de l'IoU
    iou = inter_area / tf.maximum(union_area, 1e-6)  # Évite la division par zéro

    # Perte IoU
    loss = 1.0 - iou
    loss *=10
    # Moyenne de la perte IoU
    return tf.reduce_mean(loss,axis=1)

def custom_loss(alpha=2.0, beta=1.0, gamma=0.5):
    def combined_loss(y_true, y_pred):
        # Clip predictions to valid range [0, 1]
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # 1. Modified IoU Loss with stability improvements
        def iou_loss_component(y_true, y_pred):
            # Extract coordinates
            x_min_true, y_min_true, x_max_true, y_max_true = tf.split(y_true, 4, axis=-1)
            x_min_pred, y_min_pred, x_max_pred, y_max_pred = tf.split(y_pred, 4, axis=-1)
            
            # Ensure proper ordering of coordinates
            x_min_true, x_max_true = tf.minimum(x_min_true, x_max_true), tf.maximum(x_min_true, x_max_true)
            y_min_true, y_max_true = tf.minimum(y_min_true, y_max_true), tf.maximum(y_min_true, y_max_true)
            x_min_pred, x_max_pred = tf.minimum(x_min_pred, x_max_pred), tf.maximum(x_min_pred, x_max_pred)
            y_min_pred, y_max_pred = tf.minimum(y_min_pred, y_max_pred), tf.maximum(y_min_pred, y_max_pred)
            
            # Calculate intersection
            inter_x_min = tf.maximum(x_min_true, x_min_pred)
            inter_y_min = tf.maximum(y_min_true, y_min_pred)
            inter_x_max = tf.minimum(x_max_true, x_max_pred)
            inter_y_max = tf.minimum(y_max_true, y_max_pred)
            
            # Calculate areas with small epsilon to prevent zero areas
            epsilon = 1e-7
            inter_area = tf.maximum(epsilon, inter_x_max - inter_x_min) * \
                        tf.maximum(epsilon, inter_y_max - inter_y_min)
            
            true_area = tf.maximum(epsilon, x_max_true - x_min_true) * \
                       tf.maximum(epsilon, y_max_true - y_min_true)
            pred_area = tf.maximum(epsilon, x_max_pred - x_min_pred) * \
                       tf.maximum(epsilon, y_max_pred - y_min_pred)
            
            union_area = true_area + pred_area - inter_area
            
            # Calculate IoU with improved numerical stability
            iou = inter_area / (union_area + epsilon)
            iou = tf.clip_by_value(iou, epsilon, 1.0 - epsilon)
            
            return 1.0 - iou
        
        # 2. Smooth L1 Loss (Huber Loss) for coordinates
        def coord_loss_component(y_true, y_pred):
            diff = tf.abs(y_true - y_pred)
            less_than_one = tf.cast(diff <= 1.0, tf.float32)
            smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + \
                           (1.0 - less_than_one) * (diff - 0.5)
            return smooth_l1_loss

        # 3. Modified size consistency loss with better stability
        def size_consistency_loss(y_true, y_pred):
            _, _, x_max_pred, y_max_pred = tf.split(y_pred, 4, axis=-1)
            x_min_pred, y_min_pred, _, _ = tf.split(y_pred, 4, axis=-1)
            
            width_pred = tf.maximum(0.0, x_max_pred - x_min_pred)
            height_pred = tf.maximum(0.0, y_max_pred - y_min_pred)
            
            # Softer penalties for size
            min_size = 0.05  # Minimum expected size ratio
            max_size = 0.95  # Maximum expected size ratio
            
            size_penalty = tf.reduce_mean(
                tf.maximum(0.0, min_size - width_pred) + 
                tf.maximum(0.0, width_pred - max_size) +
                tf.maximum(0.0, min_size - height_pred) + 
                tf.maximum(0.0, height_pred - max_size)
            )
            
            return size_penalty

        # Combine losses with weighted contributions
        iou_component = tf.reduce_mean(iou_loss_component(y_true, y_pred))
        coord_component = tf.reduce_mean(coord_loss_component(y_true, y_pred))
        size_component = tf.reduce_mean(size_consistency_loss(y_true, y_pred))
        
        # Final loss with gradient clipping
        total_loss = alpha * iou_component + beta * coord_component + gamma * size_component
        return tf.clip_by_value(total_loss, -100, 100)  # Prevent extreme values
    
    return combined_loss

def get_training_config():
    return {
        'optimizer': tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.001,
                first_decay_steps=1000
            )
        ),
        'batch_size': 32,  # Smaller batch size for better generalization
        'callbacks': [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="face/model/checkpoints/model_{epoch:02d}_{val_loss:.4f}.keras",
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]
    }


if __name__ == "__main__":
    # Create generators
    train_gen = data_generator(
        os.path.join(IMAGES_DIR, "train"),
        LABELS_DIR,
        batch_size=BATCH_SIZE
    )
    val_gen = data_generator(
        os.path.join(IMAGES_DIR, "val"),
        LABELS_DIR,
        batch_size=BATCH_SIZE
    )

    train_image_count = len(os.listdir(os.path.join(IMAGES_DIR, "train")))
    val_image_count = len(os.listdir(os.path.join(IMAGES_DIR, "val")))

    steps_per_epoch = train_image_count // BATCH_SIZE
    validation_steps = val_image_count // BATCH_SIZE





    checkpoint_callback = ModelCheckpoint(
        filepath="face/model/checkpoints/epoch_{epoch:02d}.keras",
        save_weights_only=False,
        #save_best_only=True,  # Changed to save only best model
        monitor='val_loss',   # Monitor validation loss
        #mode='min',          # Save when loss is minimized
        verbose=1
    )

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    # Get training configuration
    config = get_training_config()

    # Create and compile model
    model = create_improved_face_model((IMG_SIZE[0], IMG_SIZE[1], 3))
    model.compile(
        optimizer=config['optimizer'],
        loss=custom_loss(alpha=1.0, beta=2.0, gamma=1.0),
        metrics=['mse', 'mae']
    )

    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=10,
        batch_size=config['batch_size'],
        callbacks=config['callbacks']
    )

    model.save("face/face_detection_model.keras")

    # Optional: Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MAE')
    plt.plot(history.history['val_mse'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()
