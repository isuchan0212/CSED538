import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from dataset import get_cifar10_dataset
from train_val_test import train_val, test
from model import Net
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
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

def objective(trial):
    # Define the model
    model = Net().to(device)
    
    # Define the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD'])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Define the learning rate scheduler
    scheduler_name = args.lr_scheduler
    
    if scheduler_name == 'step':
        step_size = trial.suggest_int('step_size', 10, 50, step=10)
        gamma = trial.suggest_float('gamma', 0.1, 0.9, step=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'exponential':
        gamma = trial.suggest_float('gamma', 0.1, 0.9, step=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'polynomial':
        power = trial.suggest_int('power', 1, 5)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=power)
    elif scheduler_name == 'cosine':
        T_max = trial.suggest_int('T_max', 10, 50, step=10)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == 'cosinewarmup':
        T_0 = trial.suggest_int('T_0', 10, 50, step=10)
        T_mult = trial.suggest_int('T_mult', 1, 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    elif scheduler_name == "linear":
        lambda1 = lambda epoch: 1 - epoch / num_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        scheduler = None

    train_accuracies, val_accuracies, logs = train_val(model, device, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler)
    test_accuracy = test(model, device, test_loader)
    
    # We are interested in validation accuracy, so we return the highest validation accuracy
    return max(val_accuracies)

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Save the best trial
    trial = study.best_trial
    print(f'Best trial value: {trial.value}')
    print('Best hyperparameters: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Train the final model with the best hyperparameters
    best_model = Net().to(device)
    best_optimizer = optim.SGD(best_model.parameters(), lr=trial.params['lr'])
    best_scheduler = None

    if args.lr_scheduler == 'step':
        best_scheduler = torch.optim.lr_scheduler.StepLR(best_optimizer, step_size=trial.params['step_size'], gamma=trial.params['gamma'])
    elif args.lr_scheduler == 'exponential':
        best_scheduler = torch.optim.lr_scheduler.ExponentialLR(best_optimizer, gamma=trial.params['gamma'])
    elif args.lr_scheduler == 'polynomial':
        best_scheduler = torch.optim.lr_scheduler.PolynomialLR(best_optimizer, total_iters=num_epochs, power=trial.params['power'])
    elif args.lr_scheduler == 'cosine':
        best_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(best_optimizer, T_max=trial.params['T_max'])
    elif args.lr_scheduler == 'cosinewarmup':
        best_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(best_optimizer, T_0=trial.params['T_0'], T_mult=trial.params['T_mult'])
    elif args.lr_scheduler == "linear":
        lambda1 = lambda epoch: 1 - epoch / num_epochs
        best_scheduler = torch.optim.lr_scheduler.LambdaLR(best_optimizer, lr_lambda=lambda1)

    train_accuracies, val_accuracies, logs = train_val(best_model, device, num_epochs, train_loader, val_loader, criterion, best_optimizer, best_scheduler)
    test_accuracy = test(best_model, device, test_loader)

    # Update test accuracy in logs
    for log in logs:
        log['test_accuracy'] = test_accuracy

    # Save logs to CSV
    keys = logs[0].keys()
    with open('tuning_training_logs_step.csv', 'w', newline='') as output_file:
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
    plt.savefig('tuning_training_accuracy.png', bbox_inches='tight')
    plt.show()
