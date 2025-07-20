
m3_Zero-Shot-Classification-CLIP.ipynb
m3_Zero-Shot-Classification-CLIP.ipynb_
1.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Notebook Overview: Zero Shot Image Classification using CLIP
In this notebook, we will learn about Zero-Shot Image Classification using a Vision Language Model (VLM) called CLIP. Let's break down what that means.

Image Classification is a class of techniques where the input is an image and the output is a single class label from a collection of pre-defined classes. For example, given an image of a dog, an image classifier would output the label "dog."
Alt Image Classification

Zero Shot refers to the fact that we will not need any training data to build the classifier.

CLIP (Contrastive Language-Image Pre-training) is a multimodal model developed by OpenAI that learns to associate images with their textual descriptions, enabling tasks like zero-shot classification and image retrieval.

We'll start by installing some relevant libraries.

2.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Imports

[ ]
# Import necessary libraries
from PIL import Image # Used for image processing
import requests # Used for downloading images from URLs (if needed)
from transformers import CLIPProcessor, CLIPModel # Import CLIP processor and model from Hugging Face Transformers
import torch # Import PyTorch

# Use to display image
from IPython.display import display
3.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Image Classification with CLIP
This code uses a model called CLIP (Contrastive Language–Image Pre-training) for zero-shot learning, which means it can classify images into categories it hasn't been explicitly trained on. In our example, we want to classify an image as either a "cat", "dog", or "rabbit".

3.1. Load Model
We will load a version of CLIP openai/clip-vit-base-patch32. It has the following characteristics.

Architecture: Combines Vision Transformer (ViT) with a text transformer.
Purpose: Aligns images and text into a shared embedding space for multimodal tasks.
Training Data: Trained on 400M image-text pairs scraped from the internet.
Model size: Approximately 82M. Easily fits in a consumer grade GPU.
Applications: Image classification, zero-shot learning, image retrieval, and image captioning.

[ ]
# Load the pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

3.2. Load CLIP Preprocessor
The CLIP processor, loaded with the following line:

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
has a crucial role in preparing input data (both images and text labels) for the CLIP model.

Internally the CLIPProcessor applies the following set of preprocessing steps to the batch of images and text tokens.

CLIPProcessor:
- image_processor: CLIPImageProcessor {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}

- tokenizer: CLIPTokenizerFast(name_or_path='openai/clip-vit-base-patch32', vocab_size=49408, model_max_length=77, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
    49406: AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
    49407: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
Here's a breakdown of what the processor does:
1. Image Processing
Resizing: Adjusts images to a consistent size expected by the model (e.g., 224x224 pixels).
Center Cropping: Crops the central portion of the image, ensuring the primary subject remains the focal point.
Data Type Conversion: Converts images into PyTorch tensors, the format required by CLIP.
Normalization: Scales pixel values to a standard range, allowing the model to process data efficiently and consistently.
2. Text Processing
Tokenization: Splits text labels into individual tokens (words or sub-words).
Padding and Truncation: Ensures all text sequences are of uniform length by padding shorter sequences with special tokens or truncating longer sequences. Controlled via the padding=True argument.
Data Type Conversion: Converts text tokens into PyTorch tensors using special embeddings compatible with the CLIP model.
3. Combining Inputs
The processor combines the processed image and text tensors into a single dictionary-like object (inputs). This object is then passed directly to the CLIP model for prediction.

Why is the CLIP Processor important?
Consistency: It standardizes inputs, ensuring the model receives data in an optimal format for accurate and generalizable predictions.
Efficiency: Preprocessing saves computational resources by performing necessary data transformations once rather than repeatedly during inference.
Abstraction: It simplifies data preparation steps, allowing users to focus on high-level tasks instead of preprocessing complexities.

[ ]
# Load processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
3.3. Load Image & define classes

[ ]
# This section is for loading the image
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Example image URL, you can replace it with your image URL
image = Image.open(requests.get(url, stream=True).raw) # Load image from the URL

# Display the image
display(image)
3.4. Define ouput classes

[ ]
# Define the list of target labels/categories
text = ["cat", "dog", "rabbit"]
3.5. Convert raw data to model inputs

[ ]
# Preprocess the text and image using the CLIP processor
# return_tensors="pt" specifies to return PyTorch tensors
# padding=True ensures inputs are padded to the same length
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

Let's understand inputs
inputs.input_ids
This output indicates that your input_ids tensor has the shape of 3x3. This means:

3 (rows): You have provided 3 text labels or classes to the CLIP processor (likely "cat", "dog", and "rabbit"). Each row in the input_ids tensor corresponds to one of these labels.

3 (columns): The maximum sequence length for your specific text labels, after tokenization and encoding, happens to be 3. This is unusual as usually the maximum sequence length is larger, but this size depends on the specific tokens generated for the set of words you provided.

Let's break it down further:

Tokenization: The CLIP processor first tokenizes your text labels. For example, "cat" might be tokenized into a single token, while "rabbit" might be tokenized into two.

Encoding: Each token is then assigned a numerical ID from the model's vocabulary.

Padding: If any label has fewer tokens than the maximum sequence length (3 in this case), special padding tokens are added to make all labels have the same length. This ensures that the input to the model is consistent in shape.

Note: The typical maximum sequence length for the "openai/clip-vit-base-patch32" model is 77. However, in our case, because of the small set of words chosen, the maximum sequence length ended up being just 3.


[ ]
# Unpack the inputs
input_ids = inputs.input_ids  # Tokenized and encoded text input IDs
# Print information about the unpacked inputs
print("Input IDs (shape):", input_ids.shape)
print(input_ids)
Input IDs (shape): torch.Size([3, 3])
tensor([[49406,  2368, 49407],
        [49406,  1929, 49407],
        [49406, 10274, 49407]])
inputs.attention_mask
This is a binary mask used to indicate which tokens in the input_ids are actual words and which are padding tokens. It helps the model to focus on the relevant parts of the input sequence. Values of 1 indicate valid tokens, and values of 0 indicate padding.


[ ]
attention_mask = inputs.attention_mask # Mask indicating which tokens are valid (1) vs. padding (0)
print("Attention Mask (shape):", attention_mask.shape)
print(attention_mask)
Attention Mask (shape): torch.Size([3, 3])
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
inputs.pixel_values

[ ]
pixel_values = inputs.pixel_values # Processed image pixel values
print("Pixel Values (shape):", pixel_values.shape)
print(pixel_values)

Pixel Values (shape): torch.Size([1, 3, 224, 224])
tensor([[[[ 0.5873,  0.5873,  0.6165,  ...,  0.0617,  0.0471, -0.0259],
          [ 0.5727,  0.5727,  0.6603,  ...,  0.1201,  0.0763,  0.0909],
          [ 0.5873,  0.5435,  0.6165,  ...,  0.0325,  0.1201,  0.0617],
          ...,
          [ 1.8719,  1.8573,  1.8719,  ...,  1.3902,  1.4340,  1.4194],
          [ 1.8281,  1.8719,  1.8427,  ...,  1.4486,  1.4340,  1.5070],
          [ 1.8573,  1.9011,  1.8281,  ...,  1.3756,  1.3610,  1.4486]],

         [[-1.3169, -1.3019, -1.3169,  ..., -1.4970, -1.4369, -1.4820],
          [-1.2418, -1.2718, -1.2268,  ..., -1.4369, -1.4669, -1.4519],
          [-1.2568, -1.3169, -1.2268,  ..., -1.4669, -1.4069, -1.4519],
          ...,
          [ 0.1239,  0.1089,  0.1239,  ..., -0.7016, -0.6865, -0.6865],
          [ 0.0789,  0.0939,  0.0488,  ..., -0.6565, -0.6865, -0.6115],
          [ 0.0939,  0.1089,  0.0038,  ..., -0.7766, -0.7316, -0.6115]],

         [[-0.4848, -0.4137, -0.3853,  ..., -0.9541, -0.8545, -0.8545],
          [-0.4137, -0.4706, -0.3711,  ..., -0.8119, -0.8545, -0.7834],
          [-0.3284, -0.4422, -0.3853,  ..., -0.8688, -0.8119, -0.8830],
          ...,
          [ 1.5771,  1.6482,  1.6340,  ...,  0.9088,  0.9514,  0.8945],
          [ 1.6198,  1.6055,  1.6055,  ...,  0.8661,  0.8092,  0.7950],
          [ 1.6624,  1.6766,  1.5487,  ...,  0.7950,  0.8661,  0.8519]]]])
3.6. Forward pass to get output

[ ]
# Ensure no gradients are calculated for faster inference.
with torch.no_grad():
  outputs = model(**inputs) # Pass the inputs to the model
Explanation of important output fields from CLIP
outputs.logits_per_image:
This is the most important part for image classification. It contains similarity scores between the input image and each provided text label (e.g., "cat", "dog", "rabbit"). The higher the score, the more similar the image is to that label.

outputs.text_embeds:
Numerical vectors generated by CLIP representing the meaning of each provided text label. These embeddings capture semantic relationships.

outputs.image_embeds:
A numerical vector representing visual features extracted by CLIP from the input image, capturing its essential visual characteristics.

Obtain scores for each class


[ ]
print(outputs.text_embeds.shape)
print(outputs.image_embeds.shape)

logits_per_image = outputs.logits_per_image # Extract the image-text similarity scores
print("Logits Per Image (shape):", logits_per_image.shape) #The higher the score, the more similar the image is to that label.
print(logits_per_image)

# τ   (temperature multiplier)
temperature = model.logit_scale.exp().item()
print(f"Cosine similarity is scaled by: {temperature:.3f}")
torch.Size([3, 512])
torch.Size([1, 512])
Logits Per Image (shape): torch.Size([1, 3])
tensor([[23.2766, 18.2806, 19.8593]])
Cosine similarity is scaled by: 100.000
Convert scores to probabilities using softmax
The softmax function is used to convert a vector of raw scores (often called logits) into probabilities that sum to 1. It is commonly applied in classification tasks to interpret model outputs. Letus walk through the math step-by-step.

Softmax Formula
For a vector of raw scores z=[z1,z2,...,zn], the softmax function computes the probability Pi for the i-th element as:

Pi=ezi∑nj=1ezj

Where:

zi is the raw score (logit) for the i-th class,
ezi is the exponential of that score,
∑nj=1ezj is the sum of the exponentials of all scores, acting as a normalization factor,
Pi is the resulting probability for the i-th class.

[ ]
probs = logits_per_image.softmax(dim=-1) # Apply softmax to get probabilities for each label
Print output probabilities

[ ]
# Print the probabilities for each class
for i, label in enumerate(text): # Loop through the labels and their indices
  print(f"Probability of {label}: {probs[0][i].item()}") # Print the probability for each label

# Get the predicted label
predicted_label = text[probs.argmax()] # Get the label with the highest probability
print(f"\nPredicted Label: {predicted_label}") # Print the predicted label
Probability of cat: 0.9619402885437012
Probability of dog: 0.006507652346044779
Probability of rabbit: 0.031552087515592575

Predicted Label: cat
5.󠀠󠀮󠁽󠁝󠁝󠁝󠁝 Conclusion
Make sure you have learned the following concepts

Download CLIP using transformer
How to use CLIP processor to obtain image and text embeddings
How to use obtain raw similarity scores (logits)
How to convert raw scores to probabilities using softmax.
That's all!


[ ]

Start coding or generate with AI.
Colab paid products - Cancel contracts here

