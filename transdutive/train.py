import time
import matplotlib.pyplot as plt
import torch
from torch import save
import copy

def validation_accuracy(model, data, cost_function, device):
    total = 0
    correct = 0
    results = []

    with torch.set_grad_enabled(False):
        data.to(device)
        x = model(data)[data.val_mask]
        results.append(cost_function(x, data.y[data.val_mask]))

        value, pred = torch.max(x, 1)
        total += float(x.size(0))
        correct += pred.eq(data.y[data.val_mask]).sum().item()

    return sum(results) / len(results), correct * 100. / total

def store(model, file_name):
    save(copy.deepcopy(model).state_dict(), file_name)

def train(model, device, dataset, epochs, optimizer, cost_function):
    time_start = time.time()

    accuracies = []
    training_losses = []
    validation_losses = []
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

        validation_loss, accuracy = validation_accuracy(model, dataset, cost_function, device)
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
    store(best_model, f'model_acc_{max_accuracy}_ep_{epochs}')

    plt.plot(accuracies, label='Dokładność')
    plt.legend()
    plt.savefig('accuracy.png')

    plt.clf()
    plt.plot(training_losses, label='Strata zbioru treningowego')
    plt.plot(validation_losses, label='Starta zbiorty walidacyjnego')
    plt.legend()
    plt.savefig('loss.png')

    return best_model
