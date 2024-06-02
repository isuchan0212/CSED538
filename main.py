import numpy as np
import random
import matplotlib.pyplot as plt
import csv

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataset
from train_val_test import train_val, test
from model import Net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--lr_scheduler', default=None, help='Select learning rate scheduler (step, exponential, polynomial, cosine, cosinewarmup)')
args = parser.parse_args()

random_seed = 42

generator = torch.Generator().manual_seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

model = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 200

dataset = get_cifar10_dataset(root='./data', train=True, download=False, transform=transform)
test_set = get_cifar10_dataset(root='./data', train=False, download=False, transform=transform)

train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))], generator=generator)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()

if args.lr_scheduler == 'step':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # 10, 20, 30
elif args.lr_scheduler == 'exponential':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
elif args.lr_scheduler == 'polynomial':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=2)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=3)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=1)
elif args.lr_scheduler == 'cosine':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
elif args.lr_scheduler == 'cosinewarmup':
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
elif args.lr_scheduler == "linear" :
    lambda1 = lambda epoch: 1 - epoch / num_epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = None

if __name__ == '__main__':
    train_accuracies, val_accuracies, logs = train_val(model, device, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler)
    test_accuracy = test(model, device, test_loader)

    # Update test accuracy in logs
    for log in logs:
        log['test_accuracy'] = test_accuracy

    # Save logs to CSV
    keys = logs[0].keys()
    with open('linear_ag_training_logs.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(logs)

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5)) 
    plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_ag.png', bbox_inches='tight')
    plt.show()

#NONE : Accuracy of the network on the 10000 test images: 78.64%
#step : 81.25
#exp : 54,89z
#poly : 82.41
#cos : 78.97
#cw : 82.58