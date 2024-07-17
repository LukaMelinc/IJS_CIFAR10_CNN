import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import os
from torch.optim import Adam

from Models.ResNet_modular import ResNet, ResidualBlock
from Testing import simple_test
from DataLoading.CIFAR10_loader import get_data_loader

class LightningResNet(pl.LightningModule):

    def __init__(self, learning_rate, weight_decay):
        super(LightningResNet, self).__init__()
        self.model = ResNet(ResidualBlock, [2,2,2,2])
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        CIFAR10(root='./data', train=True, download=True)
        CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        cifar_trainset = CIFAR10(root='./data', train=True, transform=train_transform)
        cifar_testset = CIFAR10(root='./data', train=False, transform=test_transform)

        train_size = int(0.8 * len(cifar_trainset))
        val_size = len(cifar_trainset) - train_size
        self.cifar_train, self.cifar_val = random_split(cifar_trainset, [train_size, val_size])
        self.cifar_test = cifar_testset

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)

from pytorch_lightning.loggers import TensorBoardLogger

learning_rates = [0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006]
weight_decays = [0, 0.01, 0.03, 0.05, 0.07, 0.1]
batch_size = 64
num_epochs = 10

cifar10_dm = Cifar10DataModule(batch_size=batch_size)

for lr in learning_rates:
    for wd in weight_decays:
        model = LightningResNet(learning_rate=lr, weight_decay=wd)

        logger = TensorBoardLogger("runs/Tor_16_7", name=f"LR_{lr}_WD_{wd}")

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            logger=logger,
        )

        trainer.fit(model, cifar10_dm)
        trainer.test(model, cifar10_dm)

    
