import torch
import torchvision
from torchvision.transforms import Normalize

from rnn import SimpleRNN, train, validate
from augmentation import TrainAug, TestAug, load_data

HEIGHT = 32
DATA_DIR = './data/CIFAR10/'

def main():
    input_dim = 96
    num_classes = 10
    model = SimpleRNN(input_dim=input_dim, hidden_dim=64, num_classes=num_classes, num_layers = 3)
    criterion = torch.nn.CrossEntropyLoss()

    # Requiring modified scheduler and decay method
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    epochs = 5
    best_acc = 0
    best_f1 = 0

    train_dataloader, val_dataloader, test_dataloader = load_data()

    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, criterion, optimizer, scheduler, epoch)
        acc_val = validate(model, val_dataloader, HEIGHT, input_dim)

        print('-' * 70)
        if best_acc < acc_val:
            best_acc = acc_val  
            print(f"| best score!! | best accuracy {best_acc:8.3f}")
            torch.save(model.state_dict(), f"{DATA_DIR}/best.pth")
        
        print(f"| end of epoch {epoch:3d} | best accuracy {acc_val:8.3f}")
        
        print('-' * 70)


if __name__ == "__main__":
    main()
