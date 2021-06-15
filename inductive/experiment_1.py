import torch
from utils import get_device
from model import Net
from test import test
from train import train
from data import get_mnist_data, get_mnist_loaders

device = get_device()
print('model')
model = Net().to(device)
print('optimizer')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print('dataset')
train_dataset, test_dataset = get_mnist_data()
print('loader')
train_loader, test_loader = get_mnist_loaders(64, train_dataset, test_dataset)

print('Starting learning')
for epoch in range(1, 31):
    train(model, device, train_loader, epoch, optimizer)
    test_acc = test(device, model, test_loader, test_dataset)
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))