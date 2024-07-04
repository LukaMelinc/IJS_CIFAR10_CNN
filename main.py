
import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
import sys
from torch.nn import ReLU, MaxPool2d, LogSoftmax
from torch.nn import Module, Conv2d, Linear
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime


# hyperparameters
num_epochs = 20
learning_rate = 0.0005
num_classes = 10 # number of classes in CIFAR10
num_channels = 3 # it is a colour image(RGB), so 3 classes, if grayscale image -> num_channels = 1
momentum = 0.9
BATCH_SIZE = 32

sample_inputs = torch.rand((1, num_channels, 32, 32))  # Adjust the size as per your model's input

# Define the transform pipeline with data augmentation for the training dataset
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize the image
])

# Define the transform pipeline for the test dataset (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

writer = SummaryWriter("runs/CIFAR")

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)  
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform) 

# init data loaders
trainDataLoader = DataLoader(cifar_trainset, shuffle=True, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(cifar_testset, batch_size=BATCH_SIZE)
trainStep = len(trainDataLoader.dataset)
testStep = len(testDataLoader.dataset)

# train on GPU, if CUDA is available, else train on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def test(model, testDataLoader, device):	# test function for testing accuracy

    model.eval()  
    pravilni = 0
    vsi = 0

   # with torch.no_grad():  # Izključi izračun gradientov za hitrejše izvajanje
    for images, labels in testDataLoader:
       #images = images.reshape(-1, 32*32).to(device) # pretvori v 1D tenzor, kar pa v resnici ne želimo, konv sloji želijo več dimenzionalne podatke
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        vsi += labels.size(0)
        pravilni += (predicted == labels).sum().item()
        
    
    acc = 100 * pravilni / vsi
    return acc



# basic layer structure
class CNN(Module):
	def __init__(self, num_channels, num_classes):
		super(CNN, self).__init__()
		# first set CONV -> RELU -> POOL
		self.conv1 = Conv2d(in_channels=num_channels, out_channels=30, kernel_size=(5, 5), stride=1, padding=0)
		self.bn1 = nn.BatchNorm2d(num_features=30)  # batch normalzation layer
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
              
        # spatial dimension after three conv layers with kernel size 4 and stride 1 equals 2
              
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
       
start_time = datetime.datetime.now()
print("Start model CNN, training started at:", {start_time})
# defining the training model
model = CNN(num_channels, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.CrossEntropyLoss()
#scheduler = StepLR(optimizer, step_size=2, gamma=0.1)	# reduce learning rate by gamma every step_size epochs
num_steps = len(trainDataLoader)
#initialize training
writer.add_graph(model, sample_inputs.to(device)) # s sample_inputs.to(device) lahko sledimo, kako se vrednosti spreminjajo v modelu


for epoch in range(num_epochs):
       losses = []
       acc = []
       for i, (images, labels) in enumerate(trainDataLoader):
              labels = labels.to(device)
              
			  # forward pass
              output = model(images)
              loss = criterion(output, labels)
              writer.add_scalar('Loss/train', loss, epoch*num_steps+1)
              # backward pass
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              #scheduler.step()
              writer.add_scalar("lr", learning_rate, epoch*num_steps+1)
              
              
			  # sprotno računanje natančnosti
              _, napovedi = output.max(1)
              stevilo_pravilnih = (napovedi == labels).sum()
              sprotni_acc = float(stevilo_pravilnih) / float(images.shape[0])
              writer.add_scalar('Sprotna natančnost', sprotni_acc, epoch * num_steps + i)
              acc.append(sprotni_acc)
              losses.append(loss.item())
              
              
              if (i+1) % 100 == 0:
           	  	print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}, acc = {sprotni_acc}')
	
     	
natančnost = test(model, testDataLoader, device)
now = datetime.datetime.now()
print(f'Natančnost: {natančnost:.2f} %, training ended at: {now}')
writer.close()

# dropout layers before fully connected layers to prevent overfitting
# weight decay
