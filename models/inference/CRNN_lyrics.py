import torch
from torch import nn
from ..basic.CNN import ConvNN
from ..basic.RNN import RNNLayer

def downsample_length(orig_length, down_sample):
    if isinstance(orig_length, torch.Tensor):
        return torch.div(orig_length - 1, down_sample, rounding_mode='floor') + 1
    else:
        return (orig_length - 1) // down_sample + 1
def get_padding(kernel_size):
    if isinstance(kernel_size, int):
        padding = kernel_size // 2
    else:
        padding = tuple(x // 2 for x in kernel_size)
    return padding

class CRNN_lyrics(nn.Module):

    def __init__(
        self,
        num_classes=72,
        input_features=128,
        input_channels=1,
        num_convs=5,
        conv_channels=[64, 32, 32, 32, 32,],
        kernel_sizes=[5, 5, 3, 3, 3,],
        dropout=0.,
        num_lstms=2,
        lstm_channels=512,
        down_sample=2,
        **args
        ):

        super().__init__()

        self.num_classes = num_classes
        self.input_features = input_features
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.down_sample = down_sample
        
        self.cnn = ConvNN(
            input_channels=input_channels,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.pooling = nn.MaxPool2d(
            kernel_size=3, 
            stride=self.down_sample, 
            padding=get_padding(3),
        )

        self.rnn = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else conv_channels[-1] * downsample_length(input_features, self.down_sample), 
            hidden_size=lstm_channels,
            dropout=dropout if i < num_lstms - 1 else 0.,
            normalize=True if i > 0 else False,
            )
            for i in range(num_lstms)
            ]
            )
        self.dense = nn.Sequential(
            nn.Linear(in_features=lstm_channels * 2, out_features=lstm_channels),
            nn.ReLU(),
            nn.Linear(in_features=lstm_channels, out_features=num_classes),
            )

    def forward(self, x):
        
        if x.ndim == 3:
            x = x.squeeze(1)
        if x.shape[-2] != self.input_features:
            raise ValueError(f"Number of input features not match! expected{self.input_features} but got {x.shape[-2]}")

        x = self.cnn(x)
        # x: (batch_size, num_channels=conv_channels, num_features, length)
        x = self.pooling(x)
        old_shape = x.shape
        x = x.reshape(old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        # x: (batch_size, channel * num_features, num_frames)

        x = x.permute(2, 0, 1)
        # x: (num_frames, batch_size, num_channels)

        x = self.rnn(x)
        x = self.dense(x)
        # x: (num_frames, batch_size, num_classes)
      
        return x