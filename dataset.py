import torchvision

def get_cifar10_dataset(root, train, download, transform):
    data = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    return data

def get_cifar100_dataset(root, train, download, transform):
    data = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
    return data