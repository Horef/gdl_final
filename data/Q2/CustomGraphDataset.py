from torch_geometric.data import InMemoryDataset


class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        # Store the list of Data objects (graphs)
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root=None)

    def len(self):
        # Return the number of graphs in the dataset
        return len(self.data_list)

    def get(self, idx):
        # Return the graph at index `idx`
        return self.data_list[idx]