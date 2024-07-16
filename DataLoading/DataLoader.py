import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loader(batch_size):
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

    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    train_size = int(0.8 * len(cifar_trainset))
    val_size = len(cifar_trainset) - train_size
    train_dataset, val_dataset = random_split(cifar_trainset, [train_size, val_size])

    trainDataLoader = DataLoader(cifar_trainset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory=True)
    testDataLoader = DataLoader(cifar_testset, batch_size=batch_size)
    valDataLoader = DataLoader(val_dataset, batch_size=	batch_size)

    return trainDataLoader, valDataLoader, testDataLoader




