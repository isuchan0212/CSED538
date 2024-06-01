import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dataset import get_cifar10_dataset
from train_val_test import train_val_test
from mlp import MLP

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--lr_scheduler')
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


    train_loader = DataLoader(train_set, batch_size = 128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size = 128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size = 128, shuffle=False)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1) 

    schedulers = {
        'linear': optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / num_epochs),
        'step': optim.lr_scheduler.StepLR(optimizer, 26, gamma=0.811),
        'exponential': optim.lr_scheduler.ExponentialLR(optimizer, 0.859),
        'polynomial': optim.lr_scheduler.PolynomialLR(optimizer, 48, 2),
        'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, 38),
        'cosinewarmup': CosineAnnealingWarmupRestarts(optimizer,42,1, 0.1, 0.001, 14, 0.499)
        }

    scheduler = schedulers[args.lr_scheduler] if args.lr_scheduler else None
    train_val_test(model, device, num_epochs, train_loader, val_loader, test_loader, criterion, optimizer, scheduler)