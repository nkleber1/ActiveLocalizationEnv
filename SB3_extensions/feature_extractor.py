import gym
import torch as th
from torch import nn
from active_localization_env import Encoder
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, feat_dim=16):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "point_cloud":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                encoder_args = {  # TODO move to args
                    'pretrain': 'active_localization_env/point_clouds/weights/lidar_16.pkl',
                    'encoder_type': 'graph_s',
                    'feat_dims': feat_dim,
                    'k': 64,
                    'dataset': 'original_uni',
                    'no_vae': False,
                }
                extractors[key] = Encoder(encoder_args)
                total_concat_size += encoder_args['feat_dims']
            elif key == "depth":
                # Run through a simple MLP
                model = nn.Sequential(
                    nn.Conv1d(subspace.shape[0], 360, kernel_size=(1,)),
                    nn.ReLU(),
                    nn.Conv1d(360, 16, kernel_size=(1,)),
                    nn.ReLU()
                )
                extractors[key] = model
                total_concat_size += 16
            elif key == "encoding":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 14)
                total_concat_size += 14

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'point_cloud':
                embedding, _, _ = extractor(observations[key])
                embedding = embedding.view(-1, 16)  # TODO make variable
                encoded_tensor_list.append(embedding)
            elif key == 'depth':
                embedding = extractor(observations[key])
                embedding = embedding.view(-1, 16)  # TODO make variable
                encoded_tensor_list.append(embedding)
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
