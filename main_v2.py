# following an example from: https://github.com/adhishthite/cifar10-optimizers

import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, LogSoftmax
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

num_channels = 3
num_classes = 10
BATCH_SIZE = 64
num_epochs = 100


sample_input = torch.rand((1, num_channels, 32, 32))

# transformacija trening in testnih slik
# trening slike horizontalno/vertikalno filpnemo, dodamo padding in random cropamo
# oboje pretvorimo v tenzor in normaliziramo
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    
        
])

# Nastavimo naključne vrednosti, ki so deterministične, se bodo lahko ponovile
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

# CUDA dostopna, nastavi device na GPU/CPU
print(f'CUDA drivers are installed and ready:', "yes" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Logging
writer = SummaryWriter("runs/CIFAR")

# Priprava datasetov, razdelitev na test/val, dataloaderji
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

train_size = int(0.8 * len(cifar_trainset))
val_size = len(cifar_trainset) - train_size
train_dataset, val_dataset = random_split(cifar_trainset, [train_size, val_size])

trainDataLoader = DataLoader(cifar_trainset, shuffle=True, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True)
testDataLoader = DataLoader(cifar_testset, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=	BATCH_SIZE)

def test(model, testDataLoader, device):
    model.eval().to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testDataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total) * 100
    return acc

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),  # Adjust input dimensions accordingly
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

criterion = nn.CrossEntropyLoss().to(device)

num_steps = len(trainDataLoader)
writer.add_graph(model, sample_input.to(device))

for epoch in range(num_epochs):
    loss = []
    acc = []

    for i , (images, labels) in enumerate(trainDataLoader):
        labels = labels.to(device)
        images = images.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #acc
        _, napovedi = output.max(1)
        stevilo_pravilnih = (napovedi == labels).sum().item()
        sprotni_acc = float(stevilo_pravilnih) / float(images.shape[0])
        writer.add_scalar('Sprotna natančnost', sprotni_acc, epoch * num_steps + i)
        acc.append(sprotni_acc)
        loss.append(loss.item())
        writer.add_scalar('Loss', loss, epoch*num_steps+1)
        #writer.add_scalar("lr",scheduler.get_last_lr() , epoch*num_steps+1)
        
        if (i+1) % 100 == 0:
            #print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}, acc = {sprotni_acc}, lr ={scheduler.get_last_lr()}')
            print(f'epoch {epoch+1}/{num_epochs}, step = {i+1}/{num_steps}, loss = {loss.item():.3f}, acc = {sprotni_acc}')
                    
    val_accuracy = test(model, valDataLoader, device)
    writer.add_scalar("Val acc", val_accuracy, epoch)
    natančnost = test(model, testDataLoader, device)
    writer.add_scalar("Test acc", natančnost, epoch)
    print(f'Validation score: {val_accuracy}, Test score: {natančnost}')
    #scheduler.step()
	
     	
natančnost = test(model, testDataLoader, device)
validacija = test(model, valDataLoader, device)
writer.close()





