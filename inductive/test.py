from utils import make_confussion_matrix

def test(device, model, test_loader, test_dataset):
    model.eval()
    correct = 0
    predicted = []
    actual = []

    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch).max(1)[1]
        correct += pred.eq(batch.y).sum().item()

        predicted.extend(list(pred.cpu().numpy()))
        actual.extend(list(batch.y.cpu().numpy()))

    print(f'Test accuracy: {correct / len(test_dataset)}')
    make_confussion_matrix(predicted, actual)