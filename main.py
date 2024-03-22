import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy
from torch.nn import ReLU, MaxPool2d, LogSoftmax
from torch.nn import Module, Conv2d, Linear
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset





cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())  
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()) 

# train on GPU, if CUDA is available, else train on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# hyperparameters
num_epochs = 10
lr = 0.01

num_classes = 10 # number of classes in CIFAR10
num_channels = 3 # it is a colour image(RGB), so 3 classes, if grayscale image -> num_channels = 1
momentum = 0.9
BATCH_SIZE = 32


# init data loaders

trainDataLoader = DataLoader(cifar_trainset, shuffle=True, batch_size=BATCH_SIZE)

testDataLoader = DataLoader(cifar_testset, batch_size=BATCH_SIZE)

trainStep = len(trainDataLoader.dataset)

testStep = len(testDataLoader.dataset)

"""for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(cifar_testset[i][0], cmap= 'gray')
plt.show()"""





"""def adding_padding(image):
    #function for adding padding to images for convolution
    transform_to_tensor = transforms.ToTensor()
    image_original = image
    image_tensor = transform_to_tensor(image_original)
    padded_image_tensor = F.pad(image_tensor, (2, 2, 2, 2), mode='constant', value=0)

    plt.imshow(padded_image_tensor.permute(1, 2, 0))  # Adjusting the channel dimension for matplotlib
    plt.show()
    return padded_image_tensor"""

# basic layer structure
class CNN(Module):
	def __init__(self, num_channels, num_classes):
		super(CNN, self).__init__()
		# first set CONV -> RELU -> POOL
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=(5, 5)) # conv layer
		self.relu1 = ReLU() #relu activation
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # pooling layer
		
        # second set CONV -> RELU -> POOL
		self.conv2 = Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=(5, 5)) # conv layer 2
		self.relu2 = ReLU() # relu layer 
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) #pooling layer 2
		



        # first and second fully connected layer with ReLU
		self.fc1 = Linear(in_features=75, out_features=500) # fully connected layer 
		self.relu3 = ReLU()
		self.fc2 = Linear(in_features=500, out_features=num_classes) # fully connected layer 2
		self.logSoftmax = LogSoftmax(dim=1) # softmax activation function
		
	def forward(self, x):  # x represents a batch of input data flowing trough the network
			# pass through the first set
			#  CONV -> RELU -> POOL
			x = self.conv1(x)
			x = self.relu1(x)
			x = self.maxpool1(x)
			
			# second pass
			# CONV -> RELU -> POOL
			x = self.conv2(x)
			x = self.relu2(x)
			x = self.maxpool2(x)
			
			# flatten the output from previous sets
			x = flatten(x, 1)  # flatten into 1D list
			x = self.fc1(x)
			x = self.relu3(x)
			
			# pass through the softmax classifier
			x = self.fc2(x)
			output = self.logSoftmax(x)
			
			return output


print("Start model CNN")
# defining the training model
model = CNN(num_channels, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
criterion = nn.CrossEntropyLoss()

#initialize training

num_steps = len(trainDataLoader)

for epoch in range(num_epochs):
       for i, (images, labels) in enumerate(trainDataLoader):
              labels = labels.to(device)
              
			  # forward pass
              output = model(images)
              loss = criterion(output, labels)
              # backward pass
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              if (i+1) % 100 == 0:
           	  	print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}')
