import torch
import torch.nn as nn


class VAEBottleneck(nn.Module):
    def __init__(self, args):
        super(VAEBottleneck, self).__init__()
        self.mu = nn.Linear(args.feat_dims*2, args.feat_dims)
        self.var = nn.Linear(args.feat_dims*2, args.feat_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # Split the result into mu and var components
        mu = self.mu(x)
        log_var = self.var(x)
        # compute std
        std = torch.exp(0.5 * log_var)
        # sample eps
        eps = torch.randn_like(std)
        # “Reparameterization” trick
        z = mu + eps * std
        z = torch.unsqueeze(z, dim=1)
        return z, mu, std
