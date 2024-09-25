from itertools import count

from torch_geometric.nn import PointNetConv, fps, radius
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MLP
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import numpy as np

class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, ball_query_radius, nn):
        super().__init__()
        self.ratio = ratio
        self.ball_query_radius = ball_query_radius
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.ball_query_radius,
            batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(
            x.size(0),
            device=batch.device
        )
        return x, pos, batch

# Defining the PointNet++ model
class PointNetPlusPlus(torch.nn.Module):
    def __init__(self,
                 set_abstraction_ratio_1, set_abstraction_ratio_2, set_abstraction_radius_1, set_abstraction_radius_2, dropout):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(set_abstraction_ratio_1, set_abstraction_radius_1, MLP([3, 64, 64, 128]))
        self.sa2_module = SetAbstraction(set_abstraction_ratio_2, set_abstraction_radius_2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)


    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)


# -- Functions for training and testing the model -- #
# Defining the training step
def train_step(epoch, total_epochs, model, train_loader, optimizer, device, is_wandb=False):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        data = next(train_loader_iter).to(device)

        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, data.y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()
        total_samples += len(data.y)

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
def test_step(epoch, total_epochs, model, test_loader, device, is_wandb=False, visualize=False):
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
    error_per_class = [0] * 10
    cloud_error_per_class = [None] * 10
    cloud_error_per_class_pred = [None] * 10
    cloud_non_error_per_class = [None] * 10

    test_loader_iter = iter(test_loader)
    for batch_idx in progress_bar:
        data = next(test_loader_iter).to(device)

        with torch.no_grad():
            prediction = model(data)

        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()
        prediction_class = prediction.max(1)[1]

        cloud_size = data.pos.size(0)//len(data.y)
        for i in range(len(prediction_class)):
            if prediction_class[i] != data.y[i]:
                error_per_class[data.y[i]] += 1
                cloud_error_per_class[data.y[i]] = data.pos[i*cloud_size:(i+1)*cloud_size].cpu().numpy()
                cloud_error_per_class_pred[data.y[i]] = prediction_class[i]
            else:
                cloud_non_error_per_class[data.y[i]] = data.pos[i*cloud_size:(i+1)*cloud_size].cpu().numpy()

        correct += prediction_class.eq(data.y).sum().item()
        total_samples += len(data.y)

    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / total_samples

    print(f'Test loss: {epoch_loss:.4f}, Test accuracy: {epoch_accuracy:.4f}')
    if is_wandb:
        wandb.log({
            "Test/Loss": epoch_loss,
            "Test/Accuracy": epoch_accuracy
        })

    # Find the least erred class, and the most erred class
    if visualize:
        class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

        print(f'Error per class: {error_per_class}')
        existing_cloud_errors = [1 if cloud is not None else 0 for cloud in cloud_error_per_class]
        print(f'Existing cloud errors {existing_cloud_errors}')
        existing_cloud_non_errors = [1 if cloud is not None else 0 for cloud in cloud_non_error_per_class]
        print(f'Existing cloud non-errors {existing_cloud_non_errors}')

        least_erred_class = error_per_class.index(min(error_per_class))
        least_erred_class_name = class_names[least_erred_class]

        most_erred_class = error_per_class.index(max(error_per_class))
        most_erred_class_name = class_names[most_erred_class]

        print(f'Least erred class: {least_erred_class} - {least_erred_class_name},\n'
              f'Most erred class: {most_erred_class} - {most_erred_class_name}')
        # For the least erred class, visualize the point cloud of non-error and error samples
        if cloud_error_per_class[least_erred_class] is not None:
            print('Exists error on least erred class')
            plot_3d_scatter(cloud_error_per_class[least_erred_class], label='least_erred_class_error',
                            title=f'Error on the Least erred on class {least_erred_class_name}.\n'
                                  f'Mistaken for {class_names[cloud_error_per_class_pred[least_erred_class]]}')
        if cloud_non_error_per_class[least_erred_class] is not None:
            print('Exists non-error on least erred class')
            plot_3d_scatter(cloud_non_error_per_class[least_erred_class], label='least_erred_class_non_error',
                            title=f'Non-Error on the Least erred on class {least_erred_class_name}')

        # For the most erred class, visualize the point cloud of non-error and error samples
        if cloud_error_per_class[most_erred_class] is not None:
            print('Exists error on most erred class')
            plot_3d_scatter(cloud_error_per_class[most_erred_class], label='most_erred_class_error',
                            title=f'Error on the Most erred on class {most_erred_class_name}.\n'
                                  f'Mistaken for {class_names[cloud_error_per_class_pred[most_erred_class]]}')
        if cloud_non_error_per_class[most_erred_class] is not None:
            print('Exists non-error on most erred class')
            plot_3d_scatter(cloud_non_error_per_class[most_erred_class], label='most_erred_class_non_error',
                            title=f'Non-Error on the Most erred on class {most_erred_class_name}')

    return epoch_loss, epoch_accuracy

def plot_3d_scatter(vec, label=None, title=None, is_wandb=False):
    '''Plot a vector data as a 3D scatter'''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vec[:, 0], vec[:, 1], vec[:, 2], s=80, label=label)
    if title:
        plt.title(title)
    else:
        plt.title(label)
    if is_wandb:
        wandb.log({f"3D Scatter {label}": wandb.Image(plt)})

    plt.savefig(f'./results/Q1/graphs/{label}.png')