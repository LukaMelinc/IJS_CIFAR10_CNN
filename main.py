import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear
from torchvision import transforms, datasets, models
import time
import matplotlib as plt


cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(cifar_testset[i][0], cmap= 'gray')
plt.show()
"""
# hyperparameters
num_epochs = 0
lr = 0
num_layers = 0
imput_size = 0
hidden_size = 0
num_classes = 0
batch_size = 0
"""

