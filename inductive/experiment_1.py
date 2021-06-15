import torch
from utils import get_device
from model import Net
from test import test
from train import train
from data import get_mnist_data, get_mnist_loaders
from torch import nn

device = get_device()
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
cost_function =  nn.CrossEntropyLoss()
train_dataset, test_dataset = get_mnist_data()
train_loader, test_loader, validation_loader = get_mnist_loaders(64, train_dataset, test_dataset)
epochs = 1

print('Starting learning')
model = train(model, device, train_loader, validation_loader, epochs, optimizer, cost_function)
print('Starting testing')
test(device, model, test_loader, test_dataset)

print('Bye, have a lovely coding day!')
