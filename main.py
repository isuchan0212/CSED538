import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataset
from train_val_test import train_val, test
from mlp import MLP

transform = transforms.Compose(
    [transforms.ToTensor()]
)
generator = torch.Generator().manual_seed(42)

model = MLP()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 100

dataset = get_cifar10_dataset(root='./data', train=True, download=False, transform=transform)
test_set = get_cifar10_dataset(root='./data', train=False, download=False, transform=transform)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)


train_loader = DataLoader(train_set, batch_size = 64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size = 64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = 64, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


if __name__ == '__main__':
    train_val(model, device, num_epochs, train_loader,val_loader, criterion, optimizer)
    test(model, device, test_loader)