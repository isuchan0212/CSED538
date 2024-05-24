import torch
import matplotlib.pyplot as plt

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
            train_correct += (predicted == targets).sum().item() * inputs.size(0)

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
                val_correct += (predicted == targets).sum().item() * inputs.size(0)

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


    
    fig = plt.figure()
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    
    if scheduler is not None:
        fig.suptitle(f'Loss when using {scheduler.__class__.__name__}')
        plt.savefig(f'figures/Loss_{scheduler.__class__.__name__}')
    else:
        if optimizer.__class__.__name__ == 'Adam':
            fig.suptitle(f'Loss when using amsgrad')
            plt.savefig(f'figures/Loss_Amsgrad')
        else:
            fig.suptitle(f'Loss when using no scheduler')
            plt.savefig(f'figures/Loss_No_Scheduler')

    fig = plt.figure()
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.plot(test_acc_list)

    if scheduler is not None:
        fig.suptitle(f'Accuracy when using {scheduler.__class__.__name__}')
        plt.savefig(f'figures/Accuracy_{scheduler.__class__.__name__}')
    else:
        if optimizer.__class__.__name__ == 'Adam':
            fig.suptitle(f'Loss when using amsgrad')
            plt.savefig(f'figures/Accuracy_Amsgrad')
        else:
            fig.suptitle(f'Loss when using no scheduler')
            plt.savefig(f'figures/Accuracy_No_Scheduler')
    


def final_test(model, device, test_loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total):.2f}%')