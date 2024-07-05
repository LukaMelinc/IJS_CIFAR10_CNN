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
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import datetime


# hyperparameters
num_epochs = 15
learning_rate = 0.001
num_classes = 10 
num_channels = 3 
momentum = 0.9
BATCH_SIZE = 64

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
# Prestavitev procesov na GPU, če je navoljo
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)	# nastavimo defaultni tensor tip
print(f'CUDA drivers are installed and ready: ',torch.cuda.is_available()) # preverimo, ali so Nvidia driverji pravilno nameščeni 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	#če je GPU navoljo, treniramo na GPU


writer = SummaryWriter("runs/CIFAR")


cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)  
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform) 

train_size = int(0.8 * len(cifar_trainset))
val_size = len(cifar_trainset) - train_size
train_dataset, val_dataset = random_split(cifar_trainset, [train_size, val_size])

trainDataLoader = DataLoader(cifar_trainset, shuffle=True, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True)
testDataLoader = DataLoader(cifar_testset, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=	BATCH_SIZE)
 

def test(model, testDataLoader, device):	# test function for testing accuracy
    model.eval().to(device)  
    pravilni = 0
    vsi = 0
   # with torch.no_grad():  # Izključi izračun gradientov za hitrejše izvajanje
    for images, labels in testDataLoader:
		#images = images.reshape(-1, 32*32).to(device) # pretvori v 1D tenzor, kar pa v resnici ne želimo, konv sloji želijo več dimenzionalne podatke
        labels = labels.to(device)
        images = images.to(device)
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
       
print("Started training the model")
model = CNN(num_channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = StepLR(optimizer, step_size=2, gamma=0.7)	# reduce learning rate by gamma every step_size epochs
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor = 0.01, total_iters=10)
num_steps = len(trainDataLoader)
writer.add_graph(model, sample_inputs.to(device)) 

for epoch in range(num_epochs):
		losses = []
		acc = []
		for i, (images, labels) in enumerate(trainDataLoader):
				labels = labels.to(device)
				images = images.to(device)
				#print(images.device)
				
				# forward pass
				output = model(images)
				loss = criterion(output, labels)
				
				# backward pass
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				
				# sprotno računanje natančnosti
				_, napovedi = output.max(1)
				stevilo_pravilnih = (napovedi == labels).sum()
				sprotni_acc = float(stevilo_pravilnih) / float(images.shape[0])
				writer.add_scalar('Sprotna natančnost', sprotni_acc, epoch * num_steps + i)
				acc.append(sprotni_acc)
				losses.append(loss.item())
				writer.add_scalar('Loss', loss, epoch*num_steps+1)
				#writer.add_scalar("lr",scheduler.get_last_lr() , epoch*num_steps+1)
				
				if (i+1) % 100 == 0:
					print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}, acc = {sprotni_acc}, lr ={scheduler.get_last_lr()}')
					#print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}, acc = {sprotni_acc}')
                           
		val_accuracy = test(model, valDataLoader, device)
		writer.add_scalar("Val acc", val_accuracy, epoch)
		natančnost = test(model, testDataLoader, device)
		writer.add_scalar("Test acc", natančnost, epoch)
		print(f'Validation score: {val_accuracy}, Test score: {natančnost}')
		#scheduler.step()
	
     	
natančnost = test(model, testDataLoader, device)
now = datetime.datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
print(f'Natančnost: {natančnost:.3f} %, training ended at: {formatted_now}')
writer.close()