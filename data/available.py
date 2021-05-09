from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from data.manipulate import UnNormalize
from data.ASL import ASL


# Specify available data-sets
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'fmnist': datasets.FashionMNIST,
    'ASL' : ASL
}


# Specify available transforms
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'fmnist': [
        transforms.ToTensor(),
    ],
    'mnist28': [
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'ASL':[
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ],
    'ASL_norm':[
        transforms.Normalize(mean=[0.5175, 0.4902, 0.5006], std=[0.1981, 0.2310, 0.2434])
    ],
    'ASL_denorm': UnNormalize(mean=[0.5175, 0.4902, 0.5006], std=[0.1981, 0.2310, 0.2434]),
    'cifar10_norm': [
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ],
    'cifar100_norm': [
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ],
    'cifar10_denorm': UnNormalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    'cifar100_denorm': UnNormalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    'augment_from_tensor': [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ],
    'augment': [
        transforms.RandomCrop(32, padding=4, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
    ],
}


# Specify configurations of available data-sets
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'fmnist': {'size': 28, 'channels': 1, 'classes': 10},
    'mnist28': {'size': 28, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'ASL':{'size':64,'channels':3,'classes':29},
}
