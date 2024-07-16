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


sample_input = torch.rand((1, num_channels, 32, 32))

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

# CUDA dostopna, nastavi device na GPU/CPU
print(f'CUDA drivers are installed and ready:', "yes" if torch.cuda.is_available() else "No")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader, val_loader, test_loader = get_data_loader(BATCH_SIZE)


model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)  
criterion = nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay=0.001, momentum=0.9)

total_step = len(train_loader)

writer.add_graph(model, sample_input.to(device))

for epoch in range(num_epochs):
    losses = []
    acc = []
    for i, (images, labels) in enumerate(train_loader):
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
        writer.add_scalar('Sprotna natančnost', sprotni_acc, epoch * total_step + i)
        writer.add_scalar('Loss', loss, epoch * total_step + i)

        if (i+1) % 100 == 0:
            print(f'epoch: [{epoch+1}/{num_epochs}], step: [{i+1}/{total_step}], loss: {loss.item():.3f}, acc: {sprotni_acc}')
    
    val_acc = simple_test(model, test_loader, device)
    natancnost = simple_test(model, val_loader, device)
    print(f'Validacija: {val_acc}, Natančnost: {natancnost}')
    writer.add_scalar('Vrednost validacije', val_acc, epoch)
    writer.add_scalar('Testni rezultat', natancnost, epoch)

val_acc = simple_test(model, test_loader, device)
natancnost = simple_test(model, val_loader, device)
print(f'Validacija: {val_acc}, Natančnost: {natancnost}')

#torch.save(model, 'ResNet_model.pth')