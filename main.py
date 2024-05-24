import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataset
from train_val_test import train_val, test
from mlp import MLP

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--lr_scheduler')
parser.add_argument('-a', '--adaptive_scheduling',action='store_true')
parser.add_argument('-e', '--epochs', type=int, default=200)

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
    [transforms.ToTensor()]
)

model = MLP()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = args.epochs

dataset = get_cifar10_dataset(root='./data', train=True, download=False, transform=transform)
test_set = get_cifar10_dataset(root='./data', train=False, download=False, transform=transform)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)


train_loader = DataLoader(train_set, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size = 128, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = 128, shuffle=False, num_workers=4, pin_memory=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9) if not args.adaptive_scheduling else optim.Adam(model.parameters(), lr=0.1, amsgrad=True)

schedulers = {
    'step': optim.lr_scheduler.StepLR(optimizer,50),
    'exponential': optim.lr_scheduler.ExponentialLR(optimizer,0.998),
    'polynomial': optim.lr_scheduler.PolynomialLR(optimizer, num_epochs),
    'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer,50),
    'cosinewarmup': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,50,T_mult=2) 
    }

scheduler = schedulers[args.lr_scheduler] if args.lr_scheduler else None

if __name__ == '__main__':
    train_val(model, device, num_epochs, train_loader,val_loader, criterion, optimizer, scheduler)
    test(model, device, test_loader)