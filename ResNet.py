# following an example from: https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

# uvozimo in odpremo knjižnice
import torch
from torch import flatten
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from torch.nn import ReLU, MaxPool2d, LogSoftmax
from torch.nn import Module, Conv2d, Linear
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import gc

# hiperparametri
num_epochs = 20
learning_rate = 0.01
num_classes = 10
num_channels = 3
BATCH_SIZE = 32

# samplanje vhodov: naključna slika, ena slika, 3 kanali, velikost 32 x 32
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

"""def data_loader(data_dir,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False):
  
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
    ])

    if test:
        dataset = datasets.CIFAR10(
          root=data_dir, train=False,
          download=True, transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


# CIFAR10 dataset 
train_loader, valid_loader = data_loader(data_dir='./data',
                                         batch_size=64)

test_loader = data_loader(data_dir='./data',
                              batch_size=64,
                              test=True)
"""
import torch

def test(model, testDataLoader, device):
    """
    Test function for evaluating model accuracy.
    
    Args:
        model: The neural network model to test.
        testDataLoader: DataLoader containing the test dataset.
        device: The device to run the model on (CPU or GPU).
    
    Returns:
        acc: The accuracy of the model on the test dataset.
    """
    model.eval().to(device)
    correct = 0
    total = 0

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for images, labels in testDataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    return acc




class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        """

        Zgradba Residual bloka.

        Prva konvolucijska plast zmanjšuje dimenzijo glede na stride,
        druga plast izvaja konvolucijo, a ohrani dimenzijo slike.

        Vhodi:
            št. vhodnih kanalov
            št. izhodnih kanalov
            stride - korak
            downsample - skip connection

        Prva konvolucijska plast dejansko lahko spremeni dimenzijo obdelane slike, če je pri klicu funkcije stride > 1 (zmanjša dimenzijo slike).
        Če je stride = 1, se pri teh parametrih dimenzija slike ohrani in se izvaja samo konvolucija.
        Druga plast ima default vrednost stride = 1, torej ne spreminja velikosti slike.
        
        """
        self.conv1 = nn.Sequential(     
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),  
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels)
                    )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):   # skip connection
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample: # če je 1, gre delat skp connection
            residual = self.downsample(x)
        out += residual # doda se skip vrednost
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        """
        
        Struktura ResNet mreže.

        1 konvolucijska plast (k=7, s=2, p=3).
        4 plasti, sestavljene iz residualnih blokov, definiranih v zgornjem classu. Št. blokov v plasti se definira ob klicu classa.
        Nakoncu avgpool (se ne rabi) in fc plast 512->10 (št. razredov v CIFAR10).

        Vhodi:
            Tip bloka: Residualni blok
            Arrray plasti/blokov: array, ki določi, koliko blokov sestavlja posamezno plast
            Št. razredov: 10
        
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3), # ven pride 16 x 16 x 64
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)   # ven pride 8 x 8 x 64
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)    # ohrani se dimenzija 8 x 8 x 64, se pa izvaja konvolucija
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)   # pride ven 4 x 4 x 128
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)   # pride ven 2 x 2 x 256
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)   # pride ven 1 x 1 x 512
        #self.avgpool = nn.AvgPool2d(7, stride=1)    #ne rabi bit kernel 7, ker je itak slika že 1x1
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        
        Sestavlja sekvence (residualnih) blokov v plasti mreže.
        Na podlagi velikosti maske, koraka (stride) in dodajanja (padding) plast spreminja ali ohranja dimenzijo obdelane slike.
        Tudi v primeru, da se dimenzije ne zmanjšajo in se samo izvaja konvolucija, se mreža uči.
        Blokom ni potrebno, da zmanjšujejo dimenzijo ali spremeniti število filtrov, takrat imajo stride/korak = 1 in so brez downsampla.

        Vhodi:
            block: Tip bloka(residual block)
            planes: št. filtrov
            blocks: število blokov v posamezni plasti
            stride: velikost koraka (v prvi konvolucijski plasti)
        
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:  # downsample se nastavi, če se zmanjšujejo dimenzije slike/št. vhodnih kanalov se ne ujema s št. izhodnih
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [] # prazen array za držanje residualnih blokov
        layers.append(block(self.inplanes, planes, stride, downsample))  # Residualni blok se ustvari in doda k layers
        self.inplanes = planes  # št. izhodnih filtrov se shrani v spremenljivko št. vhodnih filtrov

        for i in range(1, blocks):  # ustvarimo toliko layerjev, kolikor je elementov v arrayu, s katerim kličemo model
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)     

        # * je unpacking operator, odpakira elemente v listu layers, da so lahko podani kot individualni argumenti v nn.Sequential konstruktor
        # V pythonu se * uporablja pred listom/tuplom pri klicu funkcije, ko odpakira iterable in poda elemente kot ločene argumente

    def forward(self, x):   # skip connection
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)  
criterion = nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=0.001, momentum=0.9)

total_step = len(trainDataLoader)

"""for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainDataLoader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()
       

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

    # Validacija
    with torch.no_grad():
        pravilni = 0
        vsi = 0

        for images, labels in valDataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            vsi += labels.size(0)
            pravilni += (pravilni == labels).sum().item()
            del images, labels, outputs

        print('Natančnost mreže: {}%, validacija: {}%'.format(5000, 100*pravilni/vsi))

with torch.no_grad():
    pravilni = 0
    vsi = 0
    for images, labels in testDataLoader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        vsi += labels.size(0)
        pravilni += (pravilni == labels).sum().item()
        del images, labels, outputs

    print('Natančnost na testni množici: {}%'.format(10000, 100*pravilni/vsi))
"""


for epoch in range(num_epochs):
    losses = []
    acc = []
    for i, (images, labels) in enumerate(trainDataLoader):
        labels = labels.to(device)
        images = images.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #sprotno računanje natančnosti
        _, napovedi = output.max(1)
        stevilo_pravilnih = (napovedi == labels).sum()
        sprotni_acc = float(stevilo_pravilnih) / float(images.shape[0])
        acc.append(sprotni_acc)
        losses.append(loss.item())

        if (i+1) % 100 == 0:
            print(f'epoch: [{epoch+1}/{num_epochs}], step: [{i+1}/{total_step}], loss: {loss.item():.3f}, acc: {sprotni_acc}')
    
    val_acc = test(model, testDataLoader, device)
    natancnost = test(model, valDataLoader, device)
    print(f'Validacija: {val_acc}, Natančnost: {natancnost}')

val_acc = test(model, testDataLoader, device)
natancnost = test(model, valDataLoader, device)
print(f'Validacija: {val_acc}, Natančnost: {natancnost}')


torch.save(model, 'ResNet_model.pth')