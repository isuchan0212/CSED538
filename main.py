import torch
import torchvision
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt

from rnn import SimpleRNN, train, validate
from augmentation import TrainAug, TestAug, load_data

HEIGHT = 32
DATA_DIR = './data/CIFAR10/'

def main():
    input_dim = 96
    num_classes = 10
    model = SimpleRNN(input_dim=input_dim, hidden_dim=128, num_classes=num_classes, num_layers = 3)
    criterion = torch.nn.CrossEntropyLoss()

    # Requiring modified scheduler and decay method
    # 1. SGD + StepLR
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum is not necessary
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    # # 2. SGD + ExpoentialLR
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum is not necessary
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    # # 3. SGD + PolynomialLR
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum is not necessary
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 20, 0.5)

    # # 4. SGD + CosineAnnealingLR
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum is not necessary
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)

    # # 5. SGD + CosineAnnealingWarmRestarts
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # momentum is not necessary
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)

    # # 6. Adam + adaptive scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # This case, scheduler is not necessary


    epochs = 200
    best_acc = 0

    train_dataloader, val_dataloader, test_dataloader = load_data()

    train_list = []
    val_list = []
    test_list = []
    epoch_list = [epoch for epoch in range(1, epochs + 1)]

    for epoch in epoch_list:
        acc_train = train(model, train_dataloader, criterion, optimizer, scheduler, epoch)
        acc_val = validate(model, val_dataloader, HEIGHT, input_dim)
        acc_test = validate(model, test_dataloader, HEIGHT, input_dim)

        train_list.append(acc_train)
        val_list.append(acc_val)
        test_list.append(acc_test)
        
        print('-' * 70)
        if best_acc < acc_val:
            best_acc = acc_val  
            torch.save(model.state_dict(), f"{DATA_DIR}/best.pth")

        print(f"| epoch {epoch:3d} | train accuracy {acc_train:8.3f} | validation accuracy {acc_val:8.3f} | test accuracy {acc_test:8.3f}")
        # print(f"Best validation accuracy {acc_val:8.3f}")
        print('-' * 70)

    plt.plot(epoch_list, train_list, marker='o', label='Training Accuracy')
    plt.plot(epoch_list, val_list, marker='o', label='Validation Accuracy')
    plt.plot(epoch_list, test_list, marker='o', label='Test Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation and Test Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
