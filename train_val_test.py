import torch

def train_val(model, device, num_epochs, train_loader,val_loader, criterion, optimizer, scheduler=None):
    torch.multiprocessing.freeze_support()
    model.to(device)

    criterion = criterion
    optimizer = optimizer
    scheduler = scheduler

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss}')

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_running_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            print(f'Epoch {epoch+1}, Validation Loss: {val_epoch_loss}')


def test(model, device, test_loader):

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