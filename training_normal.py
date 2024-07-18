# uvozimo in odpremo knjižnice
import torch
from torch import flatten
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Models.ResNet_modular import ResNet, ResidualBlock
from Tests.simple_test import test
from DataLoading.DataLoader import get_data_loader


# hiperparametri
num_epochs = 50
learning_rate = 0.003
num_classes = 10
num_channels = 3
BATCH_SIZE = 64


writer = SummaryWriter("runs/Sre_17_7/01")

run_description = """TEstiranje ResNet z Adam (defaulten), lr=0.003, 50 epochov, brez schedulerja"""

writer.add_text("Run descriptor", run_description, 0)


sample_input = torch.rand((1, num_channels, 32, 32))

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# CUDA dostopna, nastavi device na GPU/CPU
print(f'CUDA drivers are installed and ready:', "yes" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, val_loader, test_loader = get_data_loader(BATCH_SIZE)


model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)  
criterion = nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

total_step = len(train_loader)

writer.add_graph(model, sample_input.to(device))

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
                    print(f'LR: {learning_rate}, Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

            # Validation accuracy
            val_accuracy = test(model, val_loader, device)
            test_accuracy = test(model, test_loader, device)
            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
            writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Log hyperparameters and metrics
hparams = {
    'lr': learning_rate,
    'num epoch': num_epochs,
    'batch size': BATCH_SIZE
}
metrics = {
    'accuracy': test_accuracy,
    'validation_accuracy': val_accuracy
}
writer.add_hparams(hparams, metrics)

writer.close()

print("Training complete.")
#torch.save(model, 'ResNet_model.pth')