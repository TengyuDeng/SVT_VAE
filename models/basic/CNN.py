import sys
sys.path.append("..")
from utils import get_padding

from torch import nn

class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):

        super(CNNLayer, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=get_padding(kernel_size))

    def forward(self, x):

        return self.conv(self.dropout(self.relu(self.norm(x))))


class ResCNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.2):

        super(ResCNNLayer, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        if in_channels == out_channels and stride == 1:
            self.conv1 = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=get_padding(kernel_size))

    def forward(self, x):

        x = self.conv1(self.norm(x))
        return self.conv2(self.dropout(self.relu(x))) + x

class ConvNN(nn.Module):

    def __init__(
        self,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 1],
        kernel_sizes=[5, 5, 3, 3, 3, 1],
        dropout=0.,
        **args
        ):

        super().__init__()

        if len(conv_channels) != num_convs:
            raise ValueError(f"Expect conv_channels to have {num_convs} elements but got {len(conv_channels)}!")
        if len(kernel_sizes) != num_convs:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv_channels[0], 
                kernel_size=kernel_sizes[0], 
                padding=get_padding(kernel_sizes[0]),
                ),
            *[
            ResCNNLayer(
                in_channels=conv_channels[i - 1], 
                out_channels=conv_channels[i], 
                kernel_size=kernel_sizes[i],
                dropout=dropout,
                )
            for i in range(1, num_convs)
            ],
            )

    def forward(self, x):
        
        if self.input_channels == 1 and x.ndim == 3:
            x = x.unsqueeze(-3)
        # x: (batch_size, num_channels, num_features, length)
        # tatum_frames: (batch_size, num_channels=1, num_tatums + 1)

        x = self.cnn(x)
        # x: (batch_size, num_channels=conv_channels, num_features, length)

        return x