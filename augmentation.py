import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 64
DATA_DIR = './data/CIFAR10/'


class TrainAug:
    def __init__(self, mean, std, **kwargs):
        self.transform = Compose([
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, img):
        return self.transform(img)
    
class TestAug:
    def __init__(self, mean, std, **kwargs):
        self.transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)
    
def load_data():
    train_aug = TrainAug(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    test_aug = TestAug(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    dataset = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_aug)
    test_dataset = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_aug)

    torch.manual_seed(0)
    val_ratio = 0.2
    val_dataset_size = int(len(dataset) * val_ratio)
    train_dataset_size = len(dataset) - val_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


