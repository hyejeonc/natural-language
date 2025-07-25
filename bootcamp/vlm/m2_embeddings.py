# -*- coding: utf-8 -*-
"""m2_Embeddings.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DEgvePzUQEmFkOco9GHpJl8hWxtPXFVQ

# 1.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Notebook Overview

## Background
Text and image embeddings are just vectors of numbers (say a 512x1 vector) learned by a model, that capture the gist of a sentence or a picture.

Instead of raw pixels or words, each point in the vector encodes a bit of meaning—so two text strings that “feel” similar in meaning end up with vectors that sit close together, and the same goes for visually similar images.

Contrastive Language-Image Pretraining (CLIP) enables us to create embeddings for text strings as well as images in a shared embedding space. In this shared embedding space the vector for an image, and the vector of the text description of the image will be close to one another. For example, the image of the cat and the text string "a cat" will be encoded into 512x1 vectors, and these vectors will be close to one another compared to an unrelated text string like "an airplane"

## Goals
In this notebook we willl learn how to use CLIP to do the following.
1. Learn how to obtain embedding for a text string.
2. Compare embeddings of different text strings.
3. Learn how to obtain embedding for an image.
4. Compare embeddings of different images.
5. Compare embeddings of images and text strings.

# 2.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Imports
"""

# Import necessary libraries

from transformers import CLIPTokenizer # Tokenizer for the CLIP model, converts text into tokens
from transformers import CLIPProcessor # Preprocessor for CLIP model, handles image/text preprocessing
from transformers import CLIPModel     # CLIP model for image-text embedding and similarity tasks

import torch                           # PyTorch library, used for tensor operations and GPU computations
import torch.nn.functional as F        # Contains functional API for neural network operations (e.g., activations, loss functions)
import numpy as np                     # Numerical library for array manipulation and computations

import matplotlib.pyplot as plt        # Library for plotting images and visualizations
import seaborn as sns                  # Library for creating statistical visualizations

from PIL import Image                  # Used for loading and processing images
import requests                        # Used to fetch images from URLs
from io import BytesIO                 # Enables reading binary data as file-like objects in memory

"""# 3.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Load Model

"""

# Define the model name for the CLIP variant (Vision Transformer - base, 32x32 patches)
model_name = "openai/clip-vit-base-patch32"

# Load the pre-trained CLIP model from Hugging Face
model = CLIPModel.from_pretrained(model_name)

"""# 4.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Tokenize Text Strings
We first convert the text string into *tokens.* Roughly speaking, a token is a numerical representation of a word. When a string is tokenized it is padded with start and end tokens.
"""

# Load the tokenizer associated with the specified CLIP model
tokenizer = CLIPTokenizer.from_pretrained(model_name)

# Define a list of text descriptions to embed
text = ["a donut", "a cookie", "an airplane", "a cat"]

# Tokenize and preprocess the text inputs with padding to ensure equal sequence lengths
inputs = tokenizer(text, padding=True, return_tensors="pt")

# Unpack the inputs
input_ids = inputs.input_ids  # Tokenized and encoded text input IDs

# Print information about the unpacked inputs
print("Input IDs (shape):", input_ids.shape)
print(input_ids)

"""# 5.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Find Text Embeddings
The CLIP model takes in tokenized strings and returns an 512 length embedding for every string.
"""

# Compute text embeddings without tracking gradients (inference mode)
with torch.no_grad():
    # Obtain text embeddings (feature vectors) from the CLIP model
    text_embeddings = model.get_text_features(**inputs)

# Print the shape of the resulting text embeddings tensor
# The shape is [number_of_texts, embedding_dimension]
print(text_embeddings.shape)

"""# 6.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Calculate Cosine Similarity


---


*Why did the two vectors start dating after math class?*

*Because when they calculated their cosine similarity, they realized they were practically aligned—it was love at first dot-product!*


---



**Cosine similarity** measures how similar (aligned) two vectors are by looking only at the angle between them, not their length.

* Imagine each vector as an arrow from the origin.
* The cosine of the angle \$\theta\$ between the arrows is

$$
\operatorname{cosine\_sim}(\mathbf{a}, \mathbf{b}) \;=\; \cos(\theta) \;=\;
\frac{\mathbf{a}\cdot\mathbf{b}}
     {\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}
$$

* A value of **+1** means the arrows point in exactly the same direction (perfect similarity), **0** means they’re orthogonal (no similarity), and **–1** means they point in opposite directions (complete dissimilarity).

Because it ignores magnitude, cosine similarity is ideal for comparing text or image embeddings where direction captures meaning and length may just scale with word count or pixel intensity.

The calculated cosine similarity between strings are displayed using a color coded matrix.


"""

# Text_embeddings is a tensor of shape [n, d], where:
# - n = number of text prompts
# - d = embedding dimension

# Compute the n x n cosine similarity matrix between all pairs of embeddings
# text_embeddings[:, None, :] reshapes embeddings to [n, 1, d]
# text_embeddings[None, :, :] reshapes embeddings to [1, n, d]
# cosine_similarity calculates similarity along the last dimension (d)
cosine_similarity = F.cosine_similarity(
    text_embeddings[:, None, :],    # Shape: [n, 1, d]
    text_embeddings[None, :, :],    # Shape: [1, n, d]
    dim=2                           # Calculate similarity along embedding dimension d
).cpu().numpy()                     # Move to CPU and convert tensor to NumPy array for plotting

# Initialize a matplotlib figure with specified size (width=6, height=4)
plt.figure(figsize=(6, 4))

# Create a heatmap visualization using seaborn to display the cosine similarity matrix
sns.heatmap(
    cosine_similarity,              # Matrix to visualize (n x n similarity scores)
    annot=True,                     # Annotate each cell with numeric similarity value
    fmt=".2f",                      # Format annotations to two decimal places
    cmap="coolwarm",                # Colormap for heatmap indicating negative/positive similarities
    xticklabels=text,               # Label x-axis with the original text prompts
    yticklabels=text                # Label y-axis with the original text prompts
)

# Set the plot title with font size 14
plt.title("Cosine Similarity Matrix", fontsize=14)

# Label x-axis as "Text Embeddings"
plt.xlabel("Text Embeddings")

# Label y-axis as "Text Embeddings"
plt.ylabel("Text Embeddings")

# Display the plot
plt.show()

"""# 7.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Plot Images"""

# Utility function for displaying images with labels
def plot_images(images, labels):
  n = len(images)                        # Number of images loaded successfully
  fig, axes = plt.subplots(1, n)         # Create subplots with one row and n columns
  # Loop through each subplot axis, image, and its label to display them
  for ax, img, lbl in zip(axes, images, labels):
      ax.imshow(img)                     # Display the image on the axis
      ax.set_title(lbl)                  # Set the title of the subplot to the image label
      ax.axis("off")                     # Turn off axis ticks and labels for clarity

  plt.tight_layout()                     # Adjust layout to prevent overlap
  plt.show()                             # Show the image plot

"""# 8.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Load and Display Images"""

# Load a pre-trained CLIP processor for handling images and text preprocessing
processor = CLIPProcessor.from_pretrained(model_name)

# Dictionary containing labels and their corresponding image URLs
image_urls = {
    "a donut": "https://learnopencv.com/wp-content/uploads/2025/03/donut.jpeg",
    "a cookie": "https://learnopencv.com/wp-content/uploads/2025/03/cookie.jpeg",
    "an airplane": "https://learnopencv.com/wp-content/uploads/2025/03/airplane.jpeg",
    "a cat": "https://learnopencv.com/wp-content/uploads/2025/03/cat.jpeg"
}

# Extract the list of labels from the dictionary keys
labels = list(image_urls.keys())

# Define a robust function to load images from URLs
def load_image(url):
    headers = {'User-Agent': 'Mozilla/5.0'}        # Set headers to avoid blocking by web servers
    response = requests.get(url, headers=headers)  # Request the image from the URL
    response.raise_for_status()                    # Raise an error if the download fails
    # Open the downloaded image, convert it to RGB format, and return the PIL Image
    return Image.open(BytesIO(response.content)).convert("RGB")

# Initialize empty lists for successfully loaded images and their labels
images = []

# Loop through each label to load the associated image
for label in labels:
    try:
        img = load_image(image_urls[label])  # Load image from URL
        images.append(img)                   # Append loaded image to images list
    except requests.exceptions.RequestException as e:
        # If an image fails to load, print an error message
        print(f"Failed to load {label}: {e}")

# Display the loaded images in a single row using matplotlib
plot_images(images, labels)

"""# 9.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Calculate Image Emdeddings & Display Similarity"""

# Preprocess images using CLIP processor to prepare for embedding generation
image_inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
  # Generate image embeddings using the CLIP model
  image_embeddings = model.get_image_features(**image_inputs)

# Print the shape of the resulting image embeddings tensor
# The shape is [number_of_images, embedding_dimension]
print(image_embeddings.shape)

# Compute similarity matrix
img_similarity = F.cosine_similarity(image_embeddings[:, None, :], image_embeddings[None, :, :], dim=2).cpu().numpy()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(img_similarity, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm")
plt.xlabel("Image Embeddings")
plt.ylabel("Image Embeddings")
plt.title("CLIP Image-Image Similarity Heatmap")
plt.show()

"""# 10.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Calculate Image-Text Similarity"""

# Compute similarity matrix
txt_image_similarity = F.cosine_similarity(text_embeddings[:, None, :], image_embeddings[None, :, :], dim=2).cpu().numpy()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(txt_image_similarity, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm")
plt.xlabel("Text Embeddings")
plt.ylabel("Image Embeddings")
plt.title("CLIP Image-Text Similarity Heatmap")
plt.show()

"""# 11.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Conclusion
Before you leave this notebook make sure you understand the following concepts.
1. Tokenization
2. Text embedding
3. Image embedding
4. Cosine similarity

That's all!
"""

