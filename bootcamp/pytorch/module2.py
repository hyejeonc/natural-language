# https://colab.research.google.com/drive/1cjld2XmrEYxQdFRWFM5uzSYUpTfrkO5f
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
# 미분을 구할 변수 두개를 지정한다. 
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

'''
# 만약에 함수가 선형 함수가 아니면 어떻게 될까?  =>> 된다! 
z = torch.sin(x) + torch.cos(y**2)
z.retain_grad() #By default intermediate layer weight updation is not shown.

# Compute the gradients
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}") # Gradient of x: tensor([2.5839, 7.2837])
print(f"Gradient of y: {y.grad}") # Gradient of y: tensor([ 5.5273, 32.3525])
print(f"Gradient of z: {z.grad}") # Gradient of z: tensor([1., 1.])
print(f"Result of the operation: z = {z.detach()}") # Result of the operation: z = tensor([-0.0018, -0.6583])
'''


#plt.show()
'''
##############################################
1.2. Gradient Computation Graph
A computation graph is a visual representation of the sequence of operations performed on tensors in a neural network, showing how each operation contributes to the final result. It is crucial for understanding and debugging the flow of data and gradients in deep learning models.

torchviz is a tool used to visualize the computation graph of any PyTorch model.
##############################################
'''
from torchviz import make_dot
# Visualize the computation graph
dot = make_dot(z, params={"x": x, "y": y, "z" : z})
dot.render("grad_computation_graph", format="png")

img = plt.imread("grad_computation_graph.png")
plt.imshow(img)
plt.axis('off')
plt.show()


'''
##############################################
1.3. Detaching Tensors from Computation Graph
The detach() method is used to create a new tensor that shares storage with the original tensor but without tracking operations. When you call detach(), it returns a new tensor that does not require gradients. This is useful when you want to perform operations on a tensor without affecting the computation graph.
##############################################
# gradient descent 에서 이 아래로 loss 줄이는 방법에서 최소를 구하지 못하면 어떻게 되는가? 
'''
# Let's detach z from the computation graph
print("Before detaching z from computation: ", z.requires_grad)
z_det = z.detach() # 아까 설정한 z 의 텐서를 주어진 그래프에서 떼어낸다, 
print("After detaching z from computation: ", z_det.requires_grad)

'''
##############################################
1.4. Can Backpropagation be performed when requires_grad=False?
When attempting to compute the gradients using z.backward(), a RuntimeError is raised because the tensors do not require gradients, and thus do not have a grad_fn.

In this case, since requires_grad=False was used, the computation graph is essentially empty, as no gradients will be tracked.
##############################################
# gradient descent 에서 이 아래로 loss 줄이는 방법에서 최소를 구하지 못하면 어떻게 되는가? 
'''
x = torch.tensor(2.0, requires_grad=False)
y = torch.tensor(3.0, requires_grad=False)


# Perform simple operations
z = x * y + y**2


# Compute the gradients
#z.backward()





'''
##############################################
## Question 3 만약 미분가능하지 않은 함수라면?
##############################################
'''
'''
# Create tensors with requires_grad=True 
# 미분을 구할 변수 두개를 지정한다. 
x = torch.tensor([2.0, 5.0], requires_grad=True)
y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations
z = torch.nn.ReLU(x)

z.retain_grad() #By default intermediate layer weight updation is not shown.

# Compute the gradients
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")

# Autograd will apply masking for non-differentiable operations (like ReLU, and dropout) and propagate gradients for differentiable operations. 
'''

Y = torch.tensor([1.0,], requires_grad=True)
print('torch.no_grad() == ', torch.no_grad())
with torch.no_grad():
	new_tensor = Y*2
	print(new_tensor.requires_grad, Y.requires_grad)
	
	
	
'''
##############################################
## Question 4 만약 엘레멘트가 여러개인 텐서의 backward 를 구한다면?
##############################################
'''

# Create tensors with requires_grad=True 
# 미분을 구할 변수 두개를 지정한다. 
x = torch.tensor([2.0, 5.0], requires_grad=True)
y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations
z = x + y 
print(z.sum())

z.retain_grad() #By default intermediate layer weight updation is not shown.

# Compute the gradients
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")

# Autograd will apply masking for non-differentiable operations (like ReLU, and dropout) and propagate gradients for differentiable operations. 

