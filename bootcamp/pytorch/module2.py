# Use this if you have conda installed
# !conda install -c pytorch pytorch

# Use this if you are on Google Colab
# or don't have conda installed
# !pip3 install torch

# package to visualize computation graph
# pip install -1 torchviz

# Download some digit images from MNIST dataset
#!wget -q "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg" -O "mnist_0.jpg"
#!wget -q "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg" -O "mnist_1.jpg"
# mnist는 손글씨 숫자를 제공하는 데이터 사이트이다

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # GUI 관련 백엔드이다 
import numpy as np
import cv2 # opencv
import faulthandler


print("torch version : {}".format(torch.__version__))
# 토치 버전
# torch version : 2.5.1

# Batches: Batching is a technique where multiple data samples (images, in this case) are grouped together into a single tensor. This allows efficient processing of multiple samples simultaneously, to take advantage of the parallel processing capabilities of modern hardware.
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg",cv2.IMREAD_GRAYSCALE )
digit_1_array_gray = cv2.imread("mnist_1.jpg",cv2.IMREAD_GRAYSCALE )

# Safety checks
assert digit_0_array_gray is not None, "mnist_0.jpg not found or unreadable"
assert digit_1_array_gray is not None, "mnist_1.jpg not found or unreadable"

fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].plot([1,2,3], [4,5,6])

axs[1].plot([1,2,3], [4,5,6])


axs[0].imshow(digit_0_array_og, cmap='gray',interpolation='none')
axs[0].set_title("Digit 0 Image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap="gray", interpolation = 'none')
axs[1].set_title("Digit 1 Image")
axs[1].axis('off')

#Numpy array with three channels
print("Image array shape: ",digit_0_array_og.shape)
print(f"Min pixel value:{np.min(digit_0_array_og)} ; Max pixel value : {np.max(digit_0_array_og)}")

#plt.show()

'''
##############################################
## 1. Automatic Differentiation with torch.autograd
Before proceeding autograd, will understand the basic terms:

Forward Propagation:

Computes the model's output by passing the input data through the network layers. It is often called Forward pass.
Backward Propagation:

Calculates the gradients of the loss with respect to the model's parameters using the chain rule, enabling parameter updates to minimize the loss.

1.1. torch.autograd
##############################################
'''
# Create tensors with requires_grad=True
x = torch.tensor([2.0, 5.0], requires_grad=True)
y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations
z = x * y + y**2

z.retain_grad() #By default intermediate layer weight updation is not shown.

# Compute the gradients
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")




