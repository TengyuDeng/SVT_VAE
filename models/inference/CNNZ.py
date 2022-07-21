from utils import downsample_length, get_padding

import torch
from torch import nn
from ..basic.CNN import ConvNN

ALPHA = 1e-4

class CNNZ(nn.Module):

    def __init__(
        self,
        input_features,
        output_features,
        input_channels=1,
        num_convs=5,
        conv_channels=[64, 32, 32, 32, 2],
        kernel_sizes=[5, 5, 3, 3, 1],
        dropout=0.,
        **args
        ):

        super().__init__()

        if conv_channels[-1] != 2:
            raise ValueError(f"the output must be 2 channels (mean, std), but got {conv_channels[-1]}")

        self.cnn = ConvNN(
            input_channels=input_channels,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.fc = nn.Linear(input_features, output_features)

    def forward(self, x, tatum_frames):
        if x.ndim ==  3:
            x = x.unsqueeze(1)
        output = self.fc(self.cnn(x).transpose(-1, -2))
        # output: (batch_size, num_channels=2, num_frames, output_features)
        return output[:, 0, ...], torch.abs(output[:, 1, ...]) + ALPHA
        # z_mean: (batch_size, num_frames, output_features)
        # z_std: (batch_size, num_frames, output_features)