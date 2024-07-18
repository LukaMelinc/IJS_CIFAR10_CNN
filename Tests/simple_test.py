import torch

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