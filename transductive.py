import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch import save
from torch import nn, optim, cuda
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from utils.io.save_to_file import save_to_file
import time

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#DEVICE INFO
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    print(f'Device used: {device}')
    return device


DATA_DIR = 'data/Planetoid'

MODEL_PATH = 'models'
EPOCHS = 150
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

def test(model, data, device):
    model.eval()

    data = data.to(device)
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())

    print(f'Test accuracy: {acc}')

def train(model, device, data):
    time_start = time.time()

    accuracies = []
    training_losses = []
    validation_losses = []
    max_accuracy = 0

    cel = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        cuda.empty_cache()

        data.to(device)
        pred = model(data)[data.train_mask]
        loss = cel(pred, data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_loss = float(loss)
        training_losses.append(training_loss)

        validation_loss, accuracy = validate_cel(model, data, cel, device)
        validation_losses.append(validation_loss.cpu())
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy

        print(
            f'Epoch: {epoch + 1}, Accuracy: {accuracy}%, Training loss: {training_loss}, Validation loss: {validation_loss}')

    time_end = time.time()
    print(f'Training complete. Time elapsed: {time_end - time_start}s')

    print(f'Saving best model with accuracy: {max_accuracy}')
    save_to_file(best_model, f'model_acc_{max_accuracy}_ep_{EPOCHS}')

    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()

    plt.cla()
    plt.plot(training_losses, label='Training losses')
    plt.plot(validation_losses, label='Validation losses')
    plt.legend()
    plt.show()

    return best_model

def validate_cel(model, data, cel, device):
    total = 0
    correct = 0
    results = []

    with(torch.set_grad_enabled(False)):
        data.to(device)
        x = model(data)[data.val_mask]
        results.append(cel(x, data.y[data.val_mask]))

        value, pred = torch.max(x, 1)
        total += float(x.size(0))
        correct += pred.eq(data.y[data.val_mask]).sum().item()

    return sum(results) / len(results), correct * 100. / total

def get_predicted_actual(model, data):
    predicted = []
    actual = []

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()

        predicted.extend(list(pred.numpy()))
        actual.extend(list(labels.numpy()))

    return np.array(predicted), np.array(actual)

def save_to_file(model, filename):
    try:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        save(copy.deepcopy(model).state_dict(), MODEL_PATH + '/' + filename)
        print(f'Saved model to file: {MODEL_PATH}/{filename}')
    except FileNotFoundError:
        print("Couldn't find file")

# DATASET
dataset = Planetoid(DATA_DIR, "Cora")

data = dataset[0]

device = get_device()
model = GCN(dataset).to(device)

model = train(model, device=device, data=data)

test(model, data=data, device=device)