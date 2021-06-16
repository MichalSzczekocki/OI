import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, cuda


import time

DATA_DIR = 
MODEL_PATH = 'models'
EPOCHS = 150
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4



#DEVICE INFO
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    print(f'Device used: {device}')
    return device


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


def main():
    # DATASET
    dataset = Planetoid(DATA_DIR, "Cora")
    data = dataset[0]
    device = get_device()
    model = GCN(dataset).to(device)
    model = train(model, device=device, data=data)
    test(model, data=data, device=device)

if __name__ == '__main__':
    main()
