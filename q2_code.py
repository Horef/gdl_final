import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data.Q2.CustomGraphDataset import CustomGraphDataset
import warnings
import wandb

from GAT import GAT
from GAT import train_step, val_step, test_predictions
import os

import argparse

# Parsing the arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=96)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--agg_hidden', type=int, default=32)
parser.add_argument('--fc_hidden', type=int, default=128)
parser.add_argument('--readout', type=str, default='sum')
parser.add_argument('--wandb', type=int, default=0)
args = parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    if args.wandb:
        wandb.init()

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # loading the data
    print("Loading the data")
    train_data = torch.load('data/Q2/train.pt')
    val_data = torch.load('data/Q2/val.pt')
    test_data = torch.load('data/Q2/test.pt')

    # defining the data loaders
    print("Defining the data loaders")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # defining the device
    print("Defining the device")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # defining the model
    print("Defining the model")
    model = GAT(n_layer=args.n_layer, agg_hidden=args.agg_hidden, fc_hidden=args.fc_hidden, readout=args.readout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_epochs = args.epochs

    # Training the model
    print("Training the model")
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(1, total_epochs+1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        train_loss, train_accuracy = train_step(epoch, total_epochs, model, train_loader, optimizer, device,
                                                is_wandb=args.wandb)
        val_loss, val_accuracy = val_step(epoch, total_epochs, model, val_loader, device,
                                          is_wandb=args.wandb)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    if not args.wandb:
        # Checking that the directory for graphs exists
        if not os.path.exists('./results/Q2/graphs'):
            os.makedirs('./results/Q2/graphs')

        print("Saving the graphs")

        # Plotting the training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./results/Q2/graphs/loss.png')

        # Plotting the training and validation accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./results/Q2/graphs/accuracy.png')

        # Making predictions on the test set
        print("Making predictions on the test set")
        test_predictions(model=model, test_loader=test_loader, device=device, file_name='results/Q2/predications.csv')
    else:
        wandb.log({
            'Val/Min_Loss': min(val_losses),
            'Val/Max_Accuracy': max(val_accuracies)
        })
        wandb.finish()