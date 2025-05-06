import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import cv2

def integrated_gradients(
    model,
    img_path,
    target_class=None,
    baseline=None,
    num_steps=50,
    batch_size=32
):
    """
    Compute Integrated Gradients attribution for a given image.
    
    Args:
        model: Trained tensorflow model
        img_path: Path to input image
        target_class: Index of the target class (if None, uses predicted class)
        baseline: Baseline image (if None, uses black image)
        num_steps: Number of steps for gradient computation
        batch_size: Batch size for parallel processing
        
    Returns:
        tuple: (preprocessed_image, attribution_map, integrated_gradients_overlay)
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(96, 96))
    img_array = image.img_to_array(img)
    preprocessed = np.expand_dims(img_array, axis=0) / 255.0

    # Create baseline (black image) if none provided
    if baseline is None:
        baseline = np.zeros_like(preprocessed)

    # Get prediction if target class not specified
    if target_class is None:
        prediction = model.predict(preprocessed)
        target_class = np.argmax(prediction[0])

    # Generate alphas for interpolation
    alphas = tf.linspace(0.0, 1.0, num_steps)

    # Generate interpolated images
    interpolated_images = []
    for alpha in alphas:
        interpolated_image = baseline + alpha * (preprocessed - baseline)
        interpolated_images.append(interpolated_image)
    interpolated_images = tf.concat(interpolated_images, axis=0)

    # Compute gradients in batches
    gradients = []
    for i in range(0, len(interpolated_images), batch_size):
        batch = interpolated_images[i:i + batch_size]
        with tf.GradientTape() as tape:
            tape.watch(batch)
            predictions = model(batch)
            outputs = predictions[:, target_class]
        batch_gradients = tape.gradient(outputs, batch)
        gradients.append(batch_gradients)
    
    gradients = tf.concat(gradients, axis=0)

    # Calculate integral using trapezoidal rule
    gradients = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = tf.reduce_mean(gradients, axis=0)
    
    # Scale attributions
    attribution_map = integrated_gradients * (preprocessed - baseline)
    attribution_map = tf.reduce_sum(tf.abs(attribution_map), axis=-1)
    
    # Normalize attribution map
    attribution_map = (attribution_map - tf.reduce_min(attribution_map)) / \
                     (tf.reduce_max(attribution_map) - tf.reduce_min(attribution_map))
    
    # Create heatmap overlay
    heatmap = np.uint8(255 * attribution_map)
    heatmap = cv2.applyColorMap(heatmap[0], cv2.COLORMAP_VIRIDIS)
    
    # Create superimposed image
    overlay = heatmap * 0.4 + img_array
    overlay = np.clip(overlay / 255.0, 0, 1)
    
    return preprocessed, attribution_map, overlay

def visualize_integrated_gradients(img_path, model, class_names):
    """
    Generate and display the Integrated Gradients visualization.
    
    Args:
        img_path: Path to the input image
        model: Trained model
        class_names: List of class names
    """
    # Generate attribution map
    img, attribution_map, overlay = integrated_gradients(model, img_path)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(img[0])
    plt.title('Original Image')
    plt.axis('off')
    
    # Attribution map
    plt.subplot(132)
    plt.imshow(attribution_map[0], cmap='viridis')
    plt.title('Attribution Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.suptitle(f'Predicted: {class_names[predicted_class]} ({confidence:.2%})\n' + 
                'Brighter regions contributed more to the classification')
    plt.tight_layout()
    plt.show()

def analyze_with_ig(image_path, model):
    """
    Analyze an image using Integrated Gradients.
    
    Args:
        image_path: Path to the image to analyze
        model_path: Path to the trained model
    """

    
    # Class names
    class_names = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Generate visualization
    visualize_integrated_gradients(image_path, model, class_names)

# Additional utility function for comparing multiple interpretation techniques
def compare_interpretations(img_path,model):
    """
    Compare different interpretation techniques for the same image.
    
    Args:
        img_path: Path to the image
        model_path: Path to the model
    """

    class_names = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    
    # Generate interpretations with different parameters
    _, attr_map1, _ = integrated_gradients(model, img_path, num_steps=25)
    _, attr_map2, _ = integrated_gradients(model, img_path, num_steps=50)
    _, attr_map3, _ = integrated_gradients(model, img_path, num_steps=100)
    
    # Display comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(attr_map1[0], cmap='viridis')
    plt.title('25 Steps')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(attr_map2[0], cmap='viridis')
    plt.title('50 Steps')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(attr_map3[0], cmap='viridis')
    plt.title('100 Steps')
    plt.axis('off')
    
    plt.suptitle('Comparison of Different Integration Steps')
    plt.tight_layout()
    plt.show()

classes = ["anger","contempt","disgust","fear","happy","neutral","sad","surprise"]
model = tf.keras.models.load_model("feelings/final_model.keras")
analyze_with_ig("feelings/test_img/papa.jpg",model)