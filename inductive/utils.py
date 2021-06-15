import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import torch

def make_confussion_matrix(predicted, actual):
    labels = np.arange(0, 10)
    conf_matrix = pd.DataFrame(confusion_matrix(actual, predicted, labels=labels))
    cell_text = []
    for row in range(len(conf_matrix)):
        cell_text.append(conf_matrix.iloc[row])

    plt.table(cellText=cell_text, colLabels=conf_matrix.columns, rowLabels=labels, loc='center')
    plt.axis('off')
    plt.show()

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] Using {device_str}")
    return device