from utils import get_device
from data import get_planetoid_data    
from model import Net
from train import train
from test import test
import torch
from torch import nn

dataset, data = get_planetoid_data()    
device = get_device()
model = Net(dataset).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
cost_function =  nn.NLLLoss()
epochs = 150

print('Starting learning')
model = train(model, device, data, epochs, optimizer, cost_function)
print('Starting testing')
test(device, model, data)

print('Bye, have a lovely coding day!')
