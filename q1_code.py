# making the necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import torch_geometric as tg
import torch_geometric.transforms as T
import wandb
import pickle

# to parse the arguments from the command line
import argparse

from torch_geometric import device

from PointNetPlusPlus import PointNetPlusPlus
from PointNetPlusPlus import train_step, test_step
from GBNet import GBNet

# defining the transformations
pre_transform = T.NormalizeScale()

# Parsing the arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_points', type=int, default=512)
parser.add_argument('--wandb', type=int, default=0)
args = parser.parse_args()

def run_q1():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Taking the variables from the config provided by wandb
    run_config = args

    transform = T.SamplePoints(run_config.sample_points)

    # Loading the ModelNet10 dataset from torch geometric
    train = tg.datasets.ModelNet(root='./data/Q1/train/', name='10', train=True, pre_transform=pre_transform,
                                 transform=transform)

    test = tg.datasets.ModelNet(root='./data/Q1/test/', name='10', train=False, pre_transform=pre_transform,
                                transform=transform)

    if not os.path.exists('./results/Q1/graphs'):
        os.makedirs('./results/Q1/graphs')

    # defining the parameters like batch size and number of workers
    batch_size = run_config.batch_size
    num_workers = os.cpu_count()

    # defining the DataLoader for the train and test datasets
    train_loader = tg.loader.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = tg.loader.DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Defining the device, model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    #model = PointNetPlusPlus(0.5, 0.5, 0.2, 0.4, 0.5).to(device)
    model = GBNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr)
    total_epochs = run_config.epochs

    # Training the model
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch}/{total_epochs}")
        train_loss, train_accuracy = train_step(epoch, total_epochs, model, train_loader, optimizer, device, args.wandb)
        # if the last epoch, then visualize the results
        if epoch == total_epochs - 1 and not args.wandb:
            test_loss, test_accuracy = test_step(epoch, total_epochs, model, test_loader, device, args.wandb, visualize=True)
        else:
            test_loss, test_accuracy = test_step(epoch, total_epochs, model, test_loader, device, args.wandb)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    if not args.wandb:
        # Plotting the training and validation loss
        # Plotting the training and test losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./results/Q1/graphs/loss.png')

        # Plotting the training and test accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./results/Q1/graphs/accuracy.png')

if args.wandb:
    wandb.init()
run_q1()
if args.wandb:
    wandb.finish()