import torch
import matplotlib.pyplot as plt
import csv

def save_fig(msg, optimizer, scheduler, *args):

    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__

    fig = plt.figure()
    for arg in args:
        plt.plot(arg)
    
    if scheduler is not None:
        fig.suptitle(f'{msg} when using {scheduler_name}')
        plt.savefig(f'figures/{msg}_{scheduler_name}')
    else:
        if optimizer_name == 'Adam':
            fig.suptitle(f'{msg} when using amsgrad')
            plt.savefig(f'figures/{msg}_Amsgrad')
        else:
            fig.suptitle(f'{msg} when using no scheduler')
            plt.savefig(f'figures/{msg}_No_Scheduler')

def csv_writing(optimizer, scheduler, logs):

    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__

    if scheduler_name == 'NoneType':
        if optimizer_name == 'Adam':
            scheduler_name = 'Amsgrad'
        else:
            scheduler_name = 'No_Scheduler'

    with open(f'csvs/{scheduler_name}_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        for log in logs:
            writer.writerow(log)

def train_val_test(model, device, num_epochs, train_loader,val_loader, test_loader, criterion, optimizer, scheduler=None):
    torch.multiprocessing.freeze_support()
    model.to(device)

    criterion = criterion
    optimizer = optimizer
    scheduler = scheduler

    train_loss_list = []
    val_loss_list = []

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    logs = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += (predicted == targets).sum().item() 

        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_accuracy = train_correct / len(train_loader.dataset)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_accuracy)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss}')
        print(f'Epoch {epoch+1}, Train Accuracy: {epoch_accuracy}')

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (predicted == targets).sum().item() 

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_accuracy = val_correct / len(val_loader.dataset)
            val_loss_list.append(val_epoch_loss)
            val_acc_list.append(val_epoch_accuracy)
        
            print(f'Epoch {epoch+1}, Validation Loss: {val_epoch_loss}')
            print(f'Epoch {epoch+1}, Validation Accuracy: {val_epoch_accuracy}')


            test_correct = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == targets).sum().item()
            
            test_epoch_accuracy = test_correct / len(test_loader.dataset)
            test_acc_list.append(test_epoch_accuracy)

            print(f'Epoch {epoch+1}, Test Accuracy: {test_epoch_accuracy}')

        logs.append({'epoch': epoch + 1, 
                    'train_loss': epoch_loss, 'train_accuracy': epoch_accuracy,
                    'val_loss': val_epoch_loss, 'val_accuracy': val_epoch_accuracy,
                    'test_accuracy': test_epoch_accuracy})

    
    save_fig('Loss', optimizer, scheduler, train_loss_list, val_loss_list)
    save_fig('Accuracy', optimizer, scheduler, train_acc_list, val_acc_list, test_acc_list)
    
    csv_writing(optimizer, scheduler, logs)
