from torch_geometric.datasets import Planetoid

def get_planetoid_data():
    dataset = Planetoid('../data/Planetoid', "Cora")
    data = dataset[0]
    return dataset, data