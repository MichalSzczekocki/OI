import os.path as osp
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

def get_mnist_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    transform = T.Cartesian(cat=False)
    train_dataset = MNISTSuperpixels(path, True, transform=transform)
    test_dataset = MNISTSuperpixels(path, False, transform=transform)
    return train_dataset, test_dataset

def get_mnist_loaders(batch_size, train_dataset, test_dataset):
    size = len(train_dataset)
    ratio = 0.7
    train_loader = DataLoader(train_dataset[:int(size * ratio)], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(train_dataset[int(size * ratio):], batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, validation_dataloader