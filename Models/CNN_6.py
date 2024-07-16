from torch import flatten
import torch.nn as nn
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, LogSoftmax

class CNN(nn.Module):
	def __init__(self, num_channels, num_classes):
		super(CNN, self).__init__()
		# first set CONV -> RELU -> POOL
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=30, kernel_size=(5, 5), stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(num_features=30) 
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
				
		# second set CONV -> BatchNorm -> RELU -> POOL
		self.conv2 = Conv2d(in_channels=30, out_channels=60, kernel_size=(3, 3), stride=1, padding=0)
		self.bn2 = nn.BatchNorm2d(num_features=60)  
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
				
		# third set
		self.conv3 = Conv2d(in_channels=60, out_channels=120, kernel_size=(3, 3), stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(num_features=120)  
		self.relu3 = ReLU()
		self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
              
		# first and second fully connected layer with ReLU
		self.fc1 = Linear(in_features=480, out_features=480) # fully connected layer 
		self.relu4 = ReLU()
		self.fc2 = Linear(in_features=480, out_features=480) # fully connected layer 2
		self.relu5 = ReLU()
		self.fc3 = Linear(in_features=480, out_features=num_classes) # fully connected layer 3
              # outfeatures must be num_classes as it calculates the right class
		self.logSoftmax = LogSoftmax(dim=1) # softmax activation function
		
	def forward(self, x):  # x represents a batch of input data flowing trough the network
		# pass through the first set
		#  CONV -> RELU -> POOL
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		
		# second pass
		# CONV -> RELU -> POOL
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
              
		# third pass
		# CONV -> RELU -> POOL
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu3(x)
		x = self.maxpool3(x)	  
		
		# flatten the output from previous sets
		x = flatten(x, 1)  # flatten into 1D list
		x = self.fc1(x)
		x = self.relu4(x)
		
		# pass through the softmax classifier
		x = self.fc2(x)
		x = self.relu5(x)
		
		# pass through the softmax classifier
		x = self.fc3(x)
		output = self.logSoftmax(x)
		
		return output