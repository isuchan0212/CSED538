import torch

def train_val(model, device, num_epochs, train_loader, val_loader, criterion, optimizer):
    torch.multiprocessing.freeze_support()
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 전송
            # print(f'Training on device: {inputs.device}')  # 입력 데이터 디바이스 정보 출력

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.2f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 전송
                # print(f'Validation on device: {inputs.device}')  # 입력 데이터 디바이스 정보 출력
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.2f}')
            

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 데이터를 GPU로 전송
            # print(f'Testing on device: {inputs.device}')  # 입력 데이터 디바이스 정보 출력
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total):.2f}%')
