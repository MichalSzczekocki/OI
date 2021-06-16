from utils import make_confussion_matrix

def test(device, model, data,):
    model.eval()
    correct = 0
    predicted = []
    actual = []

    data = data.to(device)
    pred = model(data).max(1)[1]
    correct += pred.eq(data.y).sum().item()

    predicted.extend(list(pred.cpu().numpy()))
    actual.extend(list(data.y.cpu().numpy()))

    print(f'Test accuracy: {correct / len(data.y)}')
    make_confussion_matrix(predicted, actual)
    