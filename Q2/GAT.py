import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATv2Conv

import wandb
from tqdm import tqdm
import numpy as np
import pandas as pd

def mini_readout(x, readout):
    if readout == 'max':
        return torch.max(x, dim=0)[0]  # max readout
    elif readout == 'avg':
        return torch.mean(x, dim=0)  # avg readout
    elif readout == 'sum':
        return torch.sum(x, dim=0)  # sum readout

class GAT(nn.Module):
    def __init__(self, n_feat=7, n_class=2, n_layer=3, agg_hidden=14, fc_hidden=28,
                 heads=1, dropout=0.5, device=torch.device('cuda'), readout='sum'):
        super(GAT, self).__init__()

        self.n_layer = n_layer
        self.heads = heads
        self.dropout = dropout
        self.readout = readout
        self.device = device

        # Graph attention layer
        self.graph_attention_layers = []
        self.graph_attention_layers.append(GATv2Conv(in_channels=n_feat, out_channels=agg_hidden, dropout=dropout,
                                                     edge_dim=4))
        for i in range(self.n_layer-1):
            self.graph_attention_layers.append(GATv2Conv(in_channels=agg_hidden, out_channels=agg_hidden, dropout=dropout,
                                                         edge_dim=4))

        for i in range(self.n_layer):
            self.graph_attention_layers[i].to(device)

        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, x, edge_index, edge_attr):
        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph attention layer
        #x = torch.cat([F.relu(att(x, edge_index, edge_attr)) for att in self.graph_attention_layers], dim=2)
        for i in range(self.n_layer):
            x = F.relu(self.graph_attention_layers[i](x, edge_index, edge_attr))

            if i != self.n_layer-1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout
        x = mini_readout(x, self.readout)

        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x

    def __repr__(self):
        layers = ''

        for i in range(self.n_layer):
            layers += str(self.graph_attention_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers

def train_step(epoch, total_epochs, model, train_loader, optimizer, device, is_wandb=False):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)

    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{total_epochs}"
    )
    total_samples = 0

    train_loader_iter = iter(train_loader)
    for batch_idx in progress_bar:
        graph = next(train_loader_iter).to(device)

        optimizer.zero_grad()
        prediction = model(graph.x, graph.edge_index, graph.edge_attr)
        weight = torch.tensor([2, 1], dtype=torch.float32).to(device)
        loss = F.cross_entropy(prediction, graph.y[0], weight=weight)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted_class = torch.argmax(prediction)
        correct += predicted_class.eq(graph.y).sum().item()
        total_samples += len(graph.y)

    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / total_samples

    print(f'Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.4f}')
    if is_wandb:
        wandb.log({
            "Train/Loss": epoch_loss,
            "Train/Accuracy": epoch_accuracy
        })

    return epoch_loss, epoch_accuracy


# Defining the evaluation step
def val_step(epoch, total_epochs, model, test_loader, device, is_wandb=False):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()

    epoch_loss, correct = 0, 0
    num_val_examples = len(test_loader)

    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch}/{total_epochs}"
    )
    total_samples = 0

    test_loader_iter = iter(test_loader)
    for batch_idx in progress_bar:
        graph = next(test_loader_iter).to(device)

        with torch.no_grad():
            prediction = model(graph.x, graph.edge_index, graph.edge_attr)

        weight = torch.tensor([2, 1], dtype=torch.float32).to(device)
        loss = F.cross_entropy(prediction, graph.y[0], weight=weight)
        epoch_loss += loss.item()

        predicted_class = torch.argmax(prediction)
        correct += predicted_class.eq(graph.y).sum().item()
        total_samples += len(graph.y)

    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / total_samples

    print(f'Validation loss: {epoch_loss:.4f}, Validation accuracy: {epoch_accuracy:.4f}')
    if is_wandb:
        wandb.log({
            "Val/Loss": epoch_loss,
            "Val/Accuracy": epoch_accuracy
        })

    return epoch_loss, epoch_accuracy

def test_predictions(model, test_loader, device, file_name):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.eval()

    num_test_examples = len(test_loader)

    progress_bar = tqdm(
        range(num_test_examples),
        desc="Making predictions on test set"
    )

    # creating a dataframe to store the predictions
    prediction_df = pd.DataFrame(columns=['label', 'score'])

    test_loader_iter = iter(test_loader)
    for batch_idx in progress_bar:
        graph = next(test_loader_iter).to(device)

        with torch.no_grad():
            prediction = model(graph.x, graph.edge_index, graph.edge_attr)

        predicted_class = torch.argmax(prediction)
        predicted_sureness = torch.max(prediction)

        prediction_df.loc[batch_idx] = [predicted_class.item(), predicted_sureness.item()]

    prediction_df.to_csv(file_name, index=False)