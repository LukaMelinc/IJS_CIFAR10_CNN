# uvozimo in odpremo knjižnice
import torch
from torch import flatten
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Models.ResNet_modular import ResNet, ResidualBlock
from Testing import simple_test
from DataLoading.DataLoader import get_data_loader


# hiperparametri
num_epochs = 10
learning_rate = 0.01
num_classes = 10
num_channels = 3
BATCH_SIZE = 64

writer = SummaryWriter("runs/Tor_16_7/01")

run_description = """
num epoch: 10
Testiranje lr in weight decay za Adam
"""

writer.add_text("Run descriptor", run_description, 0)


# samplanje vhodov: naključna slika, ena slika, 3 kanali, velikost 32 x 32
sample_input = torch.rand((1, num_channels, 32, 32))

# Nastavimo naključne vrednosti, ki so deterministične, se bodo lahko ponovile
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

# CUDA dostopna, nastavi device na GPU/CPU
print(f'CUDA drivers are installed and ready:', "yes" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Logging


train_loader, val_loader, test_loader = get_data_loader(BATCH_SIZE)


learning_rates = [0.002, 0.001, 0.0009, 0.0008, 0.0007, 0.0006]
weight_decays = [0, 0.01, 0.03, 0.05, 0.07, 0.1]

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss().to(device)
# Main training loop

for lr in learning_rates:
    for wd in weight_decays:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
        
        writer = SummaryWriter(f"runs/Tor_16_7/01/_LR_{lr}_wd_{wd}")

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
            val_accuracy = simple_test(model, val_loader, device)
            test_accuracy = simple_test(model, test_loader, device)
            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
            writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Log hyperparameters and metrics
        hparams = {
            'lr': lr,
            'Weight_decay': wd,
            
        }
        metrics = {
            'accuracy': test_accuracy,
            'validation_accuracy': val_accuracy
        }
        writer.add_hparams(hparams, metrics)

        writer.close()

print("Training complete.")