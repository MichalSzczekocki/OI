import time
import matplotlib.pyplot as plt
import torch
from torch import save
import copy

def validation_accuracy(model, validation_loader, cost_function, device):
    total = 0
    correct = 0
    results = []

    model.eval()

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
    
    for epoch in range(epochs):
        model.train()

        losses = []
        correct = 0
        total = 0
        for batch in train_loader:
            batch.to(device)
            pred = model(batch)
            loss = cost_function(pred, batch.y)
            losses.append(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, labels = torch.max(pred, 1)
            correct += labels.eq(batch.y).sum().item()
            total += len(batch.y)

        training_accuracy = correct * 100 / total
        accuracies.append(training_accuracy)

        training_loss = float(sum(losses) / len(losses))
        training_losses.append(training_loss)


        validation_loss, validation_acc = validation_accuracy(model, validation_loader, cost_function, device)
        validation_losses.append(validation_loss.cpu())
        

        if validation_acc > max_accuracy:
            best_model = model
            max_accuracy = validation_accuracy

        print(
            f'Epoch: {epoch + 1}, Training accuracy {training_accuracy} Validataion Accuracy: {validation_acc}%, Training loss: {training_loss}, Validation loss: {validation_loss}')

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
