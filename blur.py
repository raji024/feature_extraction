import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load an image using PIL
try:
    # Make sure you have an image named "catsample.jpg" in the same directory
    image = Image.open("catsample.jpg").convert("L") # Convert to grayscale
except FileNotFoundError:
    print("Error: 'catsample.jpg' not found. Creating a dummy image.")
    image = Image.fromarray(np.random.randint(0, 256, (256, 256), dtype=np.uint8))

image = image.resize((256, 256)) # Resize for simplicity
image_np = np.array(image, dtype=np.float32) / 255.0 # Normalize to range [0,1]

# Add batch and channel dimensions (1, 256, 256, 1)
image_tensor = np.expand_dims(image_np, axis=(0, -1))

# Define a 3x3 blurring filter (averaging kernel)
blur_filter = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]], dtype=np.float32)

# Reshape the filter for TensorFlow (H, W, in_channels, out_channels)
blur_filter = blur_filter.reshape(3, 3, 1, 1)

# Apply convolution for blurring
blurred_image = tf.nn.conv2d(image_tensor, blur_filter, strides=[1, 1, 1, 1], padding="SAME")

# Convert tensor to numpy for visualization
blurred_image_np = blurred_image.numpy().squeeze()

# Plot original and blurred images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image_tensor.squeeze(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred_image_np, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")

plt.show() # Display the plot