import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import Adam
from pytorch_lightning.loggers import TensorBoardLogger
import torch.optim.lr_scheduler as lr_scheduler

from DataLoading.DataLoader import get_data_loader
from Models.ResNet_modular import ResNet, ResidualBlock



class LightningResNet(pl.LightningModule):
    def __init__(self, learning_rate, scheduler_name):
        super(LightningResNet, self).__init__()
        self.model = ResNet(ResidualBlock, [2,2,2,2])
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.scheduler_name = scheduler_name

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = outputs.max(1)   # predicted je tenzor z predvidenim indeksom razreda
        correct = (predicted == labels).sum().item()    # .item() metoda iz tenzorja potegne vrednost in jo pretvori v floar/int
        accuracy = correct / labels.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        if self.scheduler_name == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif self.scheduler_name == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        elif self.scheduler_name == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        else:
            scheduler = None 

        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  # This is necessary for ReduceLROnPlateau
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_loader, self.val_loader, self.test_loader = get_data_loader(self.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

learning_rates = [0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008]
scheduler_names = ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]

batch_size = 64
num_epochs = 15
num_workers = 4

cifar10_dm = Cifar10DataModule(batch_size=batch_size, num_workers=num_workers)

for scheduler in scheduler_names:
    for lr in learning_rates:

        # Initialize the model with the current set of hyperparameters
        model = LightningResNet(learning_rate=lr, scheduler_name = scheduler)

        # Initialize the TensorBoard logger
        logger = TensorBoardLogger("runs/Cet_18_7/05", name=f"LR_{lr}_SCH_{scheduler}")

        # Hyperparameters logging
        hyperparameters = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'scheduler_name': scheduler
        }
        logger.log_hyperparams(hyperparameters)

        # Initialize the PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,
            logger=logger
        )

        # Train the model
        trainer.fit(model, cifar10_dm)
        
        # Test the model
        trainer.test(model, datamodule=cifar10_dm)