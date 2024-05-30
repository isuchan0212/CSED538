import numpy as np
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataset
from mlp import MLP

import argparse

import optuna
from optuna.trial import TrialState
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_train_examples = 128 * 30
n_valid_examples = 128 * 10

num_epochs = args.epochs

dataset = get_cifar10_dataset(root='./data', train=True, download=False, transform=transform)
test_set = get_cifar10_dataset(root='./data', train=False, download=False, transform=transform)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)


train_loader = DataLoader(train_set, batch_size = 128, shuffle=True)
val_loader = DataLoader(val_set, batch_size = 128, shuffle=False)
test_loader = DataLoader(test_set, batch_size = 128, shuffle=False)


criterion = nn.CrossEntropyLoss()


schedulers = {
    'step': 'StepLR',
    'exponential': 'ExponentialLR',
    'polynomial': 'PolynomialLR',
    'cosine': 'CosineAnnealingLR',
    'cosinewarmup':'CosineAnnealingWarmRestarts'
    }


def objective(trial):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1) 
    scheduler_name = schedulers[args.lr_scheduler]
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 10, 50)
        gamma = trial.suggest_float('gamma', 0.1, 0.9, log=True)
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer,step_size,gamma)
    elif scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 0.9, log=True)
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer,gamma)
    elif scheduler_name == 'PolynomialLR':
        total_iters = trial.suggest_int('total_iters', 10, 50)
        power = trial.suggest_int('power',1,5)
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer,total_iters,power)
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 10, 50)
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer,T_max)
    else:
        first_cycle_steps = trial.suggest_int('first_cycle_steps', 20, 50)
        cycle_mult = trial.suggest_int('cycle_mult', 1, 4)
        min_lr = trial.suggest_float('min_lr', 0.001, 0.1, log=True)
        warmup_steps = trial.suggest_int('warmup_steps', 10, 20)
        gamma = trial.suggest_float('gamma', 0.1, 0.9, log=True)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps,cycle_mult, 0.1, min_lr, warmup_steps, gamma)

    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * 128 >= n_train_examples:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Limiting validation data.
                if batch_idx * 128 >= n_valid_examples:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # Get the index of the max log-probability.
                
                correct += (predicted == targets).sum().item() 

        accuracy = correct / min(len(val_loader.dataset), n_valid_examples)
        
        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy}')

        trial.report(accuracy, epoch)

        scheduler.step() 


        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=36000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))