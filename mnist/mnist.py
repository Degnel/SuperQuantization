import torch
import torch.nn as nn
import torch.optim as optim
from mnist.mnist_quantized_net import MNISTQuantizedNet
from full_net import FullPrecisionNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mnist():
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    batch_size = 8
    depth_quant = 3

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MNISTQuantizedNet(depth_quant, input_dim, hidden_dim, output_dim)
    train_model(model, train_loader)
    test_model(model, test_loader)


def train_model(model, train_loader, epochs=2, lr=0.001, grad_clamp=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda' if torch.cuda.is_available() else 'cpu'), \
                        target.to('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(data.device)
            data = data.view(data.size(0), -1)
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clamp)
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return loss.item()

# def brut_force_train(model: nn.Module, train_loader, epochs=2):
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         model.eval()
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to('cuda' if torch.cuda.is_available() else 'cpu'), \
#                         target.to('cuda' if torch.cuda.is_available() else 'cpu')
#             model = model.to(data.device)
#             data = data.view(data.size(0), -1)
#             for params in model.parameters():
#                 params.requires_grad = False
#                 for param in params.flatten():
#                     initial_outputs = model(data)
#                     initial_loss = criterion(initial_outputs, target)

#                     layer.weight.data
#                     modified_outputs = model(data)


#             if (batch_idx + 1) % 100 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda' if torch.cuda.is_available() else 'cpu'), \
                        target.to('cuda' if torch.cuda.is_available() else 'cpu')
            data = data.view(data.size(0), -1)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    return 100 * correct / total