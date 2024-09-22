from torch_geometric.nn import PointNetConv, fps, radius
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MLP
from tqdm import tqdm
import torch
import torch.nn.functional as F

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
                 set_abstraction_ratio_1, set_abstraction_ratio_2,set_abstraction_radius_1, set_abstraction_radius_2, dropout):
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
def train_step(epoch, total_epochs, model, train_loader, optimizer, device):
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)

    progress_bar = tqdm(
        range(num_train_examples),
        desc=f"Training Epoch {epoch}/{total_epochs}"
    )
    for batch_idx in progress_bar:
        data = next(iter(train_loader)).to(device)

        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, data.y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()

    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / len(train_loader.dataset)

    print(f'Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.4f}')

    return epoch_loss, epoch_accuracy


# Defining the evaluation step
def test_step(epoch, total_epochs, model, test_loader, device):
    model.eval()

    epoch_loss, correct = 0, 0
    num_val_examples = len(test_loader)

    progress_bar = tqdm(
        range(num_val_examples),
        desc=f"Validation Epoch {epoch}/{total_epochs}"
    )

    for batch_idx in progress_bar:
        data = next(iter(test_loader)).to(device)

        with torch.no_grad():
            prediction = model(data)

        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()

    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / len(test_loader.dataset)

    print(print(f'Test loss: {epoch_loss:.4f}, Test accuracy: {epoch_accuracy:.4f}'))
    return epoch_loss, epoch_accuracy