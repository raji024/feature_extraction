
import matplotlib as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load an image using PIL
# Make sure you have an image named "catsample.jpg" in the same directory
try:
    image = Image.open(r"D:\projects\feature_extraction\teddy.jpg").convert("L") # Load as grayscale
except FileNotFoundError:
    print("Error: 'catsample.jpg' not found. Please add the image to the directory.")
    # Create a dummy image to allow the rest of the script to run
    image = Image.fromarray(np.random.randint(0, 256, (256, 256), dtype=np.uint8))


image = image.resize((256, 256)) # Resize for simplicity
image_np = np.array(image, dtype=np.float32) / 255.0 # Convert to float32 and normalize

# Add batch and channel dimensions (1, 256, 256, 1)
image_tensor = np.expand_dims(image_np, axis=(0, -1))

# Define vertical and horizontal edge detection filters (Sobel-like)
vertical_filter = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]], dtype=np.float32)

horizontal_filter = np.array([[1, 1, 1],
                              [0, 0, 0],
                              [-1, -1, -1]], dtype=np.float32)

# Reshape filters for TensorFlow (H, W, in_channels, out_channels)
vertical_filter = vertical_filter.reshape(3, 3, 1, 1)
horizontal_filter = horizontal_filter.reshape(3, 3, 1, 1)

# Apply convolution
vertical_edges = tf.nn.conv2d(image_tensor, vertical_filter, strides=[1, 1, 1, 1], padding="SAME")
horizontal_edges = tf.nn.conv2d(image_tensor, horizontal_filter, strides=[1, 1, 1, 1], padding="SAME")

# Convert tensors to numpy for visualization
vertical_edges_np = vertical_edges.numpy().squeeze()
horizontal_edges_np = horizontal_edges.numpy().squeeze()

# Plot original and filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_np, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(vertical_edges_np, cmap="gray")
plt.title("Vertical Edges")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(horizontal_edges_np, cmap="gray")
plt.title("Horizontal Edges")
plt.axis("off")

plt.show()