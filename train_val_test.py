import csv
import torch

def train_val(model, device, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler=None):
    torch.multiprocessing.freeze_support()
    model.to(device)

    train_accuracies = []
    val_accuracies = []
    logs = []  # To store logs
    best_val_accuracy = 0.0  # Best validation accuracy to track improvement

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct / total
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss}, Train Accuracy: {train_accuracy:.2f}%')

        if scheduler is not None:
            scheduler.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_accuracy = 100. * correct / total
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch+1}, Validation Loss: {val_epoch_loss}, Validation Accuracy: {val_accuracy:.2f}%')

            # Save the model if the validation accuracy is the best we've seen so far.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        # Log the results
        logs.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_epoch_loss,
            'val_accuracy': val_accuracy,
            'test_accuracy': None  # Placeholder, will be updated after test phase
        })

    return train_accuracies, val_accuracies, logs

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

    test_accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {test_accuracy:.2f}%')
    return test_accuracy
