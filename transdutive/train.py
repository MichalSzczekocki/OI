import time
import matplotlib.pyplot as plt
import torch
from torch import save
import copy

def calculate_accuracy(model, data, device):
    total = 0
    correct = 0

    with torch.set_grad_enabled(False):
        data.to(device)
        x = model(data)
        _, pred = torch.max(x, 1)
        total += float(x.size(0))
        correct += pred.eq(data.y).sum().item()

    return correct * 100. / total

def store(model, file_name):
    save(copy.deepcopy(model).state_dict(), file_name)

def train(model, device, dataset, epochs, optimizer, cost_function):
    time_start = time.time()

    accuracies = []
    training_losses = []
    max_accuracy = 0
    
    model.train()
    for epoch in range(epochs):

        dataset.to(device)
        pred = model(dataset)[dataset.train_mask]
        loss = cost_function(pred, dataset.y[dataset.train_mask])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_loss = float(loss)
        training_losses.append(training_loss)

        accuracy = calculate_accuracy(model, dataset, device)
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy

        print(
            f'Epoch: {epoch + 1}, Accuracy: {accuracy}%, Training loss: {training_loss}')

    time_end = time.time()
    print(f'Training complete. Time elapsed: {time_end - time_start}s')

    print(f'Saving best model with accuracy: {max_accuracy}')
    store(best_model, f'model_acc_{max_accuracy}_ep_{epochs}')

    plt.plot(accuracies, label='Dokładność')
    plt.legend()
    plt.savefig('accuracy.png')

    plt.clf()
    plt.plot(training_losses, label='Strata zbioru treningowego')
    plt.legend()
    plt.savefig('loss.png')

    return best_model
