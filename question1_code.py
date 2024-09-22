# making the necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import torch_geometric as tg
import torch_geometric.transforms as T

from PointNetPlusPlus import PointNetPlusPlus
from PointNetPlusPlus import train_step, test_step

# defining the transformations
pre_transform = T.NormalizeScale()
transform = T.SamplePoints(2048)

if __name__ == '__main__':
    # Loading the ModelNet10 dataset from torch geometric
    train = tg.datasets.ModelNet(root='./data/Q1/train/', name='10', train=True, pre_transform=pre_transform,
                                 transform=transform)

    test = tg.datasets.ModelNet(root='./data/Q1/test/', name='10', train=False, pre_transform=pre_transform,
                                transform=transform)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # defining the parameters like batch size and number of workers
    batch_size = 64
    num_workers = os.cpu_count()

    # defining the DataLoader for the train and test datasets
    train_loader = tg.loader.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = tg.loader.DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Defining the device, model and optimizer
    device = torch.device('cpu')
    model = PointNetPlusPlus(0.5, 0.5, 0.2, 0.4, 0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total_epochs = 10

    # Training the model
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(total_epochs):
        train_loss, train_accuracy = train_step(epoch, total_epochs, model, train_loader, optimizer, device)
        test_loss, test_accuracy = test_step(epoch, total_epochs, model, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Plotting the training and validation loss
    # Plotting the training and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./results/loss.png')

    # Plotting the training and test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./results/accuracy.png')