from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "./data",
    download = False,
    train = True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root = "./data",
    download = False,
    train = False,
    transform = transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = True
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
    
model = MNISTModel()

criterion = nn.CrossEntropyLoss()

optimiser = optim.Adam(
    model.parameters(),
    lr = 0.001
)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
    
    avg_loss = running_loss/len(train_loader)

    print(
        f"epoch {epoch +1}/{epochs}"
        f"loss: {avg_loss:.4f}"
    )


correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct/total
print(f"test accuracy: {accuracy:.2f}%")