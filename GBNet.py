"""
Courtesy of:
@Author: Shi Qiu (based on Yue Wang's codes)
@Contact: shi.qiu@anu.edu.au
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sympy.core.tests.test_sympify import numpy
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1]))  # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2]))  # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st - org_x
    edge2 = neighbor_2nd - org_x
    normals = torch.cross(edge1, edge2, dim=1)  # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True)  # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True)  # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1)  # B,14,N

    return new_pts


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""

    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        point_num = 512

        self.bn1 = nn.BatchNorm1d(point_num // 8)
        self.bn2 = nn.BatchNorm1d(point_num // 8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=point_num, out_channels=point_num // 8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=point_num, out_channels=point_num // 8, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha * out + x
        return out


class ABEM_Module(nn.Module):
    """ Attentional Back-projection Edge Features Module (ABEM)"""

    def __init__(self, in_dim, out_dim, k):
        super(ABEM_Module, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, in_dim, kernel_size=[1, self.k], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(out_dim)

        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=[1, self.k], bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(out_dim)

    def forward(self, x):
        # Prominent Feature Encoding
        x1 = x  # input
        input_edge = get_graph_feature(x, k=self.k)
        x = self.conv1(input_edge)
        x2 = x  # EdgeConv for input

        x = self.conv2(x)  # LFC
        x = torch.squeeze(x, -1)
        x3 = x  # Back-projection signal

        delta = x3 - x1  # Error signal

        x = get_graph_feature(delta, k=self.k)  # EdgeConv for Error signal
        x = self.conv3(x)
        x4 = x

        x = x2 + x4  # Attentional feedback
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.caa1(x)  # B,out_dim,N

        # Fine-grained Feature Encoding
        x_local = self.conv4(input_edge)
        x_local = torch.squeeze(x_local, -1)
        x_local = self.caa2(x_local)  # B,out_dim,N

        return x, x_local


class GBNet(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=10):
        super(GBNet, self).__init__()
        self.k = k

        self.abem1 = ABEM_Module(14, 64, self.k)
        self.abem2 = ABEM_Module(64, 64, self.k)
        self.abem3 = ABEM_Module(64, 128, self.k)
        self.abem4 = ABEM_Module(128, 256, self.k)

        self.bn = nn.BatchNorm1d(emb_dims)
        self.conv = nn.Sequential(nn.Conv1d(1024, emb_dims, kernel_size=1, bias=False),
                                  self.bn,
                                  nn.LeakyReLU(negative_slope=0.2))

        self.caa = CAA_Module(1024)

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn_linear2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # x: B,3,N
        batch_size = x.size(0)

        # Geometric Point Descriptor:
        x = geometric_point_descriptor(x)  # B,14,N

        # 1st Attentional Back-projection Edge Features Module (ABEM):
        x1, x1_local = self.abem1(x)

        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        x2, x2_local = self.abem2(x1)

        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        x3, x3_local = self.abem3(x2)

        # 4th Attentional Back-projection Edge Features Module (ABEM):
        x4, x4_local = self.abem4(x3)

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        x = self.conv(x)
        x = self.caa(x)  # B,1024,N

        # global embedding
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((global_max, global_avg), 1)  # B,2048

        # FC layers with dropout
        x = F.leaky_relu(self.bn_linear1(self.linear1(x)), negative_slope=0.2)  # B,512
        x = self.dp1(x)
        x = F.leaky_relu(self.bn_linear2(self.linear2(x)), negative_slope=0.2)  # B,256
        x = self.dp2(x)
        x = self.linear3(x)  # B,C

        return x


def gb_train(model, epochs, device, opt, train_loader, test_loader, points_in_cloud:int = 512):
    criterion = cal_loss

    train_losses = []
    test_losses = []
    train_accuracies = []
    train_avg_accuracies = []
    test_accuracies = []
    test_avg_accuracies = []

    best_test_acc = 0
    for epoch in range(epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for batch in tqdm(train_loader, 'Running Batches'):
            data = batch.pos
            label = batch.y.to(device)
            batch_size = label.size()[0]
            # changing the data matrix to dimensions B, N, C, where B is the batch size, N is the number of points,
            # and C is the number of features
            data_batches = torch.split(data, points_in_cloud, dim=0)
            data = torch.stack(data_batches, dim=0)

            data = data.permute(0, 2, 1)
            data = data.to(device)

            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        train_losses.append(train_loss * 1.0 / count)
        train_accuracies.append(metrics.accuracy_score(train_true, train_pred))
        train_avg_accuracies.append(metrics.balanced_accuracy_score(train_true, train_pred))
        print(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for batch in tqdm(test_loader, desc='Running Batches'):
            data = batch.pos
            label = batch.y.to(device)
            batch_size = label.size()[0]
            # changing the data matrix to dimensions B, N, C, where B is the batch size, N is the number of points,
            # and C is the number of features
            data_batches = torch.split(data, points_in_cloud, dim=0)
            data = torch.stack(data_batches, dim=0)

            data = data.permute(0, 2, 1)
            data = data.to(device)

            with torch.no_grad():
                logits = model(data)
                loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        test_losses.append(test_loss * 1.0 / count)
        test_accuracies.append(test_acc)
        test_avg_accuracies.append(avg_per_class_acc)
        print(outstr)

    return train_losses, test_losses, train_accuracies, train_avg_accuracies, test_accuracies, test_avg_accuracies

def gb_test(model, device, test_loader, points_in_cloud:int = 512):
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    error_per_class = [0] * 10
    cloud_error_per_class = [None] * 10
    error_per_class_pred = [0] * 10
    cloud_non_error_per_class = [None] * 10

    for batch in tqdm(test_loader, desc='Running Batches'):
        data = batch.pos
        label = batch.y.to(device)
        batch_size = label.size()[0]
        # changing the data matrix to dimensions B, N, C, where B is the batch size, N is the number of points,
        # and C is the number of features
        data_batches = torch.split(data, points_in_cloud, dim=0)
        data = torch.stack(data_batches, dim=0)

        data = data.permute(0, 2, 1)
        data = data.to(device)

        with torch.no_grad():
            logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

        pred_labels = preds.cpu().numpy()
        real_labels = label.cpu().numpy()

        for i in range(len(pred_labels)):
            if pred_labels[i] != real_labels[i]:
                error_per_class[real_labels[i]] += 1
                cloud_error_per_class[real_labels[i]] = batch.pos[i*points_in_cloud:(i+1)*points_in_cloud].cpu().numpy()
                error_per_class_pred[real_labels[i]] = pred_labels[i]
            else:
                cloud_non_error_per_class[real_labels[i]] = batch.pos[i*points_in_cloud:(i+1)*points_in_cloud].cpu().numpy()

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    print(outstr)

    # Find the least erred class, and the most erred class
    class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table',
                   'toilet']

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
                              f'Mistaken for {class_names[error_per_class_pred[least_erred_class]]}')
    if cloud_non_error_per_class[least_erred_class] is not None:
        print('Exists non-error on least erred class')
        plot_3d_scatter(cloud_non_error_per_class[least_erred_class], label='least_erred_class_non_error',
                        title=f'Non-Error on the Least erred on class {least_erred_class_name}')

    # For the most erred class, visualize the point cloud of non-error and error samples
    if cloud_error_per_class[most_erred_class] is not None:
        print('Exists error on most erred class')
        plot_3d_scatter(cloud_error_per_class[most_erred_class], label='most_erred_class_error',
                        title=f'Error on the Most erred on class {most_erred_class_name}.\n'
                              f'Mistaken for {class_names[error_per_class_pred[most_erred_class]]}')
    if cloud_non_error_per_class[most_erred_class] is not None:
        print('Exists non-error on most erred class')
        plot_3d_scatter(cloud_non_error_per_class[most_erred_class], label='most_erred_class_non_error',
                        title=f'Non-Error on the Most erred on class {most_erred_class_name}')


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