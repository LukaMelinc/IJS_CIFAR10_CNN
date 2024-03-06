import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear
from torchvision import transforms#, datasets, models, ReLU, MaxPool2d, LogSoftmax
import time
import matplotlib.pyplot as plt
import numpy



#cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)



"""for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(cifar_testset[i][0], cmap= 'gray')
plt.show()"""



# hyperparameters
num_epochs = 10
lr = 0
num_layers = 0
imput_size = 0
hidden_size = 0
num_classes = 10 # number of classes in CIFAR10
num_channels = 3 # it is a colour image(RGB), so 3 classes, if grayscale image -> num_channels = 1
batch_size = 0


def adding_padding(image):
    #function for adding padding to images for convolution
    transform_to_tensor = transforms.ToTensor()
    image_original = image
    image_tensor = transform_to_tensor(image_original)
    padded_image_tensor = F.pad(image_tensor, (2, 2, 2, 2), mode='constant', value=0)

    plt.imshow(padded_image_tensor.permute(1, 2, 0))  # Adjusting the channel dimension for matplotlib
    plt.show()
    return padded_image_tensor

"""# basic layer structure
class CNN(Module):
	def __init__(self, num_channels, num_classes):
		super(CNN, self).__init__()
		# conv -> pool -> conv -> pool -> fc-> fc
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=20,kernel_size=(5, 5)) # conv layer
		self.relu1 = ReLU() #relu activation
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # pooling layer
		self.conv2 = Conv2d(in_channels=20, out_channels=50,kernel_size=(5, 5)) # conv layer 2
		self.relu2 = ReLU() # relu layer 
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) #pooling layer 2
		self.fc1 = Linear(in_features=800, out_features=500) # fully connected layer 
		self.relu3 = ReLU()
		self.fc2 = Linear(in_features=500, out_features=num_classes) # fully connected layer 2
		self.logSoftmax = LogSoftmax(dim=1) # softmax activation function"""
