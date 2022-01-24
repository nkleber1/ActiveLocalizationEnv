from .model_graph_encoder import GraphEncoderS, GraphEncoder, GraphDoubleEncoder
from .model_vae_bottleneck import VAEBottleneck
from argparse import Namespace
import torch
import torch.nn as nn
from collections import OrderedDict


def make_state_dict(pretrain):
    state_dict = torch.load(pretrain, map_location='cpu')
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key.split('.')[0] in ['encoder', 'vae_bottleneck']:
            new_state_dict[key] = val
    return new_state_dict


def select_model(args):
    encoder_model, vae_bottleneck = None, None
    if args.encoder_type == 'graph':
        encoder_model = GraphEncoder(args)
    if args.encoder_type == 'graph_s':
        encoder_model = GraphEncoderS(args)
    if args.encoder_type == 'graph_double':
        encoder_model = GraphDoubleEncoder(args)
    if not args.no_vae:
        vae_bottleneck = VAEBottleneck(args)
    return encoder_model, vae_bottleneck


class Encoder(nn.Module):
    def __init__(self, kwargs):
        super(Encoder, self).__init__()
        self.args = Namespace(**kwargs)
        self.encoder, self.vae_bottleneck = select_model(self.args)
        self.load_state_dict(make_state_dict(self.args.pretrain))
        print("Load Encoder")

    def forward(self, point_cloud):
        feature = self.encoder(point_cloud)
        mu, std = None, None
        if not self.args.no_vae:
            feature, mu, std = self.vae_bottleneck(feature)
        return feature, mu, std

    def encode_np(self, point_cloud_np):
        self.eval()
        point_cloud = torch.from_numpy(point_cloud_np)
        _, encoding, _ = self.forward(point_cloud.float())
        encoding_np = encoding.squeeze().detach().numpy()
        return encoding_np
