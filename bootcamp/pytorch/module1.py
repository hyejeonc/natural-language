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
## 1.1. Convert Numpy array to Torch tensors #
##############################################
'''
# Convert the images to PyTorch tensors and normalize
img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) / 255.0  # float64
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0 # 위의 Max pixel value 로 노멀라이즈 

print("Shape of Normalised Digit 0 Tensor: ", img_tensor_0.shape)
print(f"Normalised Min pixel value: {torch.min(img_tensor_0)} ; Normalised Max pixel value : {torch.max(img_tensor_0)}")
# 즉, (배치 크기, 채널 수, 높이, 너비)의 형태
# 배치는 모델링을 위해 분석하는 이미지의 묶음이다 


plt.imshow(img_tensor_0,cmap="gray") 
plt.title("Normalised Digit 0 Image")
plt.axis('off')
#plt.show()

'''
############################
1.2. Creating Input Batch  #
############################
'''
batch_tensor = torch.stack([img_tensor_0, img_tensor_1])
# Additionally in PyTorch, image tensors typically follow the shape convention [N ,C ,H ,W] unlike tensorflow which follows [N, H, W, C].
# In PyTorch the forward pass of input images to the model is expected to have a batch_size > 1
print("Batch Tensor Shape:", batch_tensor.shape)
# 즉 0,1 숫자 두 개의 배치, 높이, 너비, 채널수 이렇게 된다. 

batch_input = batch_tensor.permute(0,3,1,2)
print("Batch Tensor Shape:", batch_input.shape) # 파이토치는 배치, 채널, 높이, 너비로 한다 
# batch_tensor.view(5, 3, 28, 28) permute 로 안바꿔도 이렇게 할 수도 있다 
'''
####################################
# 2.1. Construct your first Tensor #
####################################
'''
# Numpy 처럼 텐서들을 다룬다. 
# Create a Tensor with just ones in a column
a = torch.ones(5)
# Print the tensor we created
print(a)

# Create a Tensor with just zeros in a column
b = torch.zeros(5)
print(b)

c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)

d = torch.zeros(3,2)
print(d)

e = torch.ones(3,2)
print(e)

f = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
print(f)

# 3D Tensor
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(g)

print(f.shape)

print(e.shape)

print(g.shape)

'''
####################################
# 2.2. Access an element in Tensor #
####################################
'''
# Get element at index 2 
# Numpy 랑 완전히 같다..! 
print(c[2])

# All indices starting from 0
# Get element at row 1, column 0
print(f[1,0])

# We can also use the following
print(f[1][0])

# Similarly for 3D Tensor
print(g[1,0,0])
print(g[1][0][0])

# All elements
print(f[:])
print('f == ', f)

# All elements from index 1 to 2 (excluding element 3)
print(c[1:3])

# All elements till index 4 (exclusive)
print(c[:4])

# First row
print(f[0, :])

# Second column
print(f[:,1])


'''
####################################
2.3. Specify data type of elements
####################################
'''
int_tensor = torch.tensor([[1,2,3],[4,5,6]])
print(int_tensor.dtype)

# What if we changed any one element to floating point number?
int_tensor = torch.tensor([[1,2,3],[4.,5,6]])
print(int_tensor.dtype)
print(int_tensor)

# This can be overridden as follows
float_tensor = torch.tensor([[1, 2, 3],[4., 5, 6]])
int_tensor = float_tensor.type(torch.int64)
print(int_tensor.dtype)
print(int_tensor)

'''
####################################
2.4. Tensor to/from NumPy Array
####################################
'''
# Tensor to Array
f_numpy = f.numpy()
print(f_numpy)

# Array to Tensor
h = np.array([[8,7,6,5],[4,3,2,1]])
h_tensor = torch.from_numpy(h)
print(h_tensor)

'''
####################################
2.5. Arithmetic Operations on Tensors
####################################
'''
# Create tensor
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])

# Addition
print(tensor1+tensor2)
# We can also use
print(torch.add(tensor1,tensor2))

# Subtraction
print(tensor1-tensor2)
# We can also use
print(torch.sub(tensor1,tensor2))

# Multiplication
# Tensor with Scalar
print(tensor1 * 2)

# Tensor with another tensor
# Elementwise Multiplication
print(tensor1 * tensor2)

# Matrix multiplication
# 행렬 곱!!!! 
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print(torch.mm(tensor1,tensor3))

# Division
# Tensor with scalar
print(tensor1/2)

# Tensor with another tensor
# Elementwise division
print(tensor1/tensor2)


'''
####################################
2.6. Broadcasting
####################################
'''

# Create two 1-dimensional tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

# adding a scalar to a vector
result = a + b

print("Result of Broadcasting:\n",result)

# Create two tensors with shapes (1, 3) and (3, 1)
# 이건 numpy 에서 잘 못보던 것 같다..! 
a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[4], [5], [6]])

# adding tensors of different shapes
result = a + b
print("Shape: ", result.shape)
print("\n")
print("Result of Broadcasting:\n", result)

'''
####################################
2.7. CPU v/s GPU Tensor
####################################
'''
# Create a tensor for CPU
# This will occupy CPU RAM
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

# Create a tensor for GPU
# This will occupy GPU RAM
# 난 nvidia gpu 램이 읎다..
#tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

# This uses CPU RAM
tensor_cpu = tensor_cpu * 5

# This uses GPU RAM
# Focus on GPU RAM Consumption
#tensor_gpu = tensor_gpu * 5

# Move GPU tensor to CPU
#tensor_gpu_cpu = tensor_gpu.to(device='cpu')

# Move CPU tensor to GPU
#tensor_cpu_gpu = tensor_cpu.to(device='cuda')

'''
####################################
Question 4
####################################
'''

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.tensor([[2], [2]])
c2 = torch.tensor([[2,2], [2,2]])
d = a+b
e = d*c
print(d)
print(e)
e2 = d*c2
print(e2) #  e는 e2 와 다르게 2x1 텐서여도, 스칼라 곱처럼 elementwise 로 각 행에 있는 텐서를 모두 스칼라곱 한다. 



