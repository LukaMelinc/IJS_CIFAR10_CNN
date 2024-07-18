import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim import Adam
from Models.ResNet_modular import ResNet, ResidualBlock
from pytorch_lightning.loggers import TensorBoardLogger

class LightningResNet(pl.LightningModule):
    def __init__(self, learning_rate):
        super(LightningResNet, self).__init__()
        self.model = ResNet(ResidualBlock, [2,2,2,2])
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_workers = num_workers

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
        return optimizer

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers = self.num_workers)

learning_rates = [0.005, 0.004, 0.003, 0.002, 0.001, 0.0009, 0.0008]

batch_size = 64
num_epochs = 12
num_workers = 4

cifar10_dm = Cifar10DataModule(batch_size=batch_size, num_workers=num_workers)

# Perform grid search for hyperparameter tuning
for lr in learning_rates:

    # Initialize the model with the current set of hyperparameters
    model = LightningResNet(learning_rate=lr)

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger("runs/Cet_18_7/01", name=f"LR_{lr}")

    # Hyperparameters logging
    hyperparameters = {
        'learning_rate': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_workers': num_workers
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