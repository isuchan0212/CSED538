# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.optim as optim
# from model import Net
# from utils import train, test, imshow, classes, adjust_learning_rate

# def main():
#     # 데이터 로드 및 전처리
#     transform = transforms.Compose(
#         [transforms.RandomHorizontalFlip(),
#          transforms.RandomCrop(32, padding=4),
#          transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     batch_size = 4

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                               shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                            download=True, transform=transforms.Compose([
#                                                transforms.ToTensor(),
#                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                            ]))
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                              shuffle=False, num_workers=2)

#     net = Net()

#     # 손실 함수 및 옵티마이저 정의
#     criterion = torch.nn.CrossEntropyLoss()
#     initial_lr = 0.01
#     optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#     # 모델 학습
#     num_epochs = 20
#     for epoch in range(num_epochs):  # 20 에포크 동안 학습
#         train(net, trainloader, criterion, optimizer, epoch)
#         scheduler.step()

#     print('Finished Training')

#     # 모델 저장
#     PATH = './cifar_net.pth'
#     torch.save(net.state_dict(), PATH)

#     # 학습 결과 시각화
#     dataiter = iter(testloader)
#     images, labels = next(dataiter)

#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#     # 모델 불러오기
#     net = Net()
#     net.load_state_dict(torch.load(PATH))

#     # 테스트 데이터셋으로 모델 평가
#     test(net, testloader)

# if __name__ == "__main__":
#     main()

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from model import SimpleVGG
from dataset import get_cifar10_dataset
from train_val_test import train_val, test

# Transformations for data augmentation
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
generator = torch.Generator().manual_seed(42)

model = SimpleVGG()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')  # 디바이스 정보 출력
model.to(device)  # 모델을 GPU로 전송

num_epochs = 20

dataset = get_cifar10_dataset(root='./data', train=True, download=True, transform=transform)
test_set = get_cifar10_dataset(root='./data', train=False, download=True, transform=transform)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    print("Training started.")
    train_val(model, device, num_epochs, train_loader, val_loader, criterion, optimizer)
    test(model, device, test_loader)
