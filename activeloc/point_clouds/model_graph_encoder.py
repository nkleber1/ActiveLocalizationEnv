"""
FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 2, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 2)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*k, 2)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 2)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
        2))  # (batch_size, num_points, 2, 1) * (batch_size, num_points, 1, 2) -> (batch_size, num_points, 2, 2)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, num_dims**2).transpose(2, 1)  # (batch_size, 4, num_points)
    x = torch.cat((pts, x), dim=1)  # (batch_size, 6, num_points)
    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x


class BaseGraphEncoder(nn.Module):
    def __init__(self, args, k=None):
        super(BaseGraphEncoder, self).__init__()
        if k:
            self.k = k
        elif args.k is None:
            self.k = 16
        else:
            self.k = args.k

        self.output_dim = args.feat_dims
        if not args.no_vae:
            self.output_dim = self.output_dim*2

        self.input_dim = 6
        if '3d' in args.dataset:
            self.input_dim *= 2

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)  # (batch_size, 2, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)  # (batch_size, 2, num_points) -> (batch_size, 6, num_points])
        x = self.mlp1(x)  # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)  # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)  # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2, 1)  # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat  # (batch_size, 1, feat_dims)


class GraphEncoder(BaseGraphEncoder):
    def __init__(self, args, k=None):
        super(GraphEncoder, self).__init__(args, k)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.output_dim, 1),
        )


class GraphEncoderS(BaseGraphEncoder):
    def __init__(self, args, k=None):
        super(GraphEncoderS, self).__init__(args, k)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.input_dim, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(32, 32)
        self.conv1 = nn.Conv1d(32, 64, 1)
        self.linear2 = nn.Linear(64, 64)
        self.conv2 = nn.Conv1d(64, 512, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, 1),
        )


class GraphDoubleEncoder(nn.Module):
    def __init__(self, args):
        super(GraphDoubleEncoder, self).__init__()
        output_dim = args.feat_dims
        if not args.no_vae:
            output_dim = output_dim * 2

        self.encoder1 = GraphEncoder(args, args.k)
        self.encoder2 = GraphEncoder(args, args.k / 4)
        self.mlp = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim, 1),
            nn.ReLU(),
        )

    def forward(self, pts):
        feature1 = self.encoder1(pts)
        feature2 = self.encoder1(pts)
        feature = torch.cat((feature1, feature2), dim=2)
        feature = feature.transpose(2, 1)
        return self.mlp(feature)
