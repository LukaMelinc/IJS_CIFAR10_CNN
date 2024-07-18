# uvozimo in odpremo knjižnice
import sys
import os
import torch
from torch import flatten
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.ResNet_modular import ResNet, ResidualBlock
from Tests.simple_test import test
from DataLoading.DataLoader import get_data_loader

#base_log_dir = os.path.join(os.path.dirname(__file__), '..', 'runs')
# hiperparametri
num_epochs = 12
learning_rate = 0.01
num_classes = 12
num_channels = 3
BATCH_SIZE = 64



writer = SummaryWriter("runs/Cet_18_7/02")
run_description = """ num epoch: 12, Testiranje lr in weight decay za Adam"""
writer.add_text("Run descriptor", run_description, 0)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
print(f'CUDA drivers are installed and ready:', "yes" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_data_loader(BATCH_SIZE)

learning_rates = [0.005, 0.004, 0.003, 0.002, 0.001, 0.0009]

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss().to(device)


for lr in learning_rates:
    
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
    writer = SummaryWriter(f"runs/Cet_18_7/02/LR_{lr}")

    # Log the model graph
    sample_input = torch.rand((1, num_channels, 32, 32)).to(device)
    writer.add_graph(model, sample_input)

    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_accuracies = []

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy)

            # Log loss and accuracy
            writer.add_scalar('Train/Loss', loss.item(), epoch * total_step + i)
            writer.add_scalar('Train/Accuracy', accuracy, epoch * total_step + i)

            if (i + 1) % 100 == 0:
                print(f'LR: {lr}, Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

        # Validation accuracy
        val_accuracy = test(model, val_loader, device)
        test_accuracy = test(model, test_loader, device)
        writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Log hyperparameters and metrics
    hparams = {
        'lr': lr
        
    }
    metrics = {
        'accuracy': test_accuracy,
        'validation_accuracy': val_accuracy
    }
    writer.add_hparams(hparams, metrics)

    writer.close()

print("Training complete.")