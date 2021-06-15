import time
import matplotlib.pyplot as plt
import torch
from torch import save
import copy

def validation_accuracy(model, validation_loader, cost_function, device):
    total = 0
    correct = 0
    results = []

    with torch.set_grad_enabled(False):
        for batch in validation_loader:
            batch.to(device)
            x = model(batch)
            results.append(cost_function(x, batch.y))

            value, pred = torch.max(x, 1)
            total += float(x.size(0))
            correct += pred.eq(batch.y).sum().item()

    return sum(results) / len(results), correct * 100. / total

def store(model, file_name):
    save(copy.deepcopy(model).state_dict(), file_name)

def train(model, device, train_loader, validation_loader, epochs, optimizer, cost_function):
    time_start = time.time()

    accuracies = []
    training_losses = []
    validation_losses = []
    max_accuracy = 0
    
    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            batch.to(device)
            pred = model(batch)
            loss = cost_function(pred, batch.y)
            losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        training_loss = float(sum(losses) / len(losses))
        training_losses.append(training_loss)

        validation_loss, accuracy = validation_accuracy(model, validation_loader, cost_function, device)
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

    plt.cla()
    fig = plt.gcf()
    plt.legend()
    plt.plot(accuracies, label='Dokładność')
    fig.savefig('model_accuracy')

    plt.cla()
    fig = plt.gcf()
    plt.legend()
    plt.plot(training_losses, label='Strata zbioru treningowego')
    plt.plot(validation_losses, label='Starta zbiorty walidacyjnego')
    fig.savefig('loss_functions')

    return best_model
