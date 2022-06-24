import sys
sys.path.append("..")

from torch import nn
from torch_scatter import segment_csr
from ..basic.CNN import ConvNN
from ..basic.TCN import TemporalConvNet


class ConvTCN(nn.Module):

    def __init__(
        self,
        num_classes_pitch,
        input_features=80,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 1],
        kernel_sizes=[5, 5, 3, 3, 3, 1],
        tcn_channels=[128, 128, 128],
        tcn_kernel=2,
        dropout=0.,
        **args
        ):

        super().__init__()

        if len(conv_channels) != num_convs:
            raise ValueError(f"Expect conv_channels to have {num_convs} elements but got {len(conv_channels)}!")
        if len(kernel_sizes) != num_convs:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")
        self.num_classes_pitch = num_classes_pitch
        self.input_features = input_features
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        
        self.cnn = ConvNN(
            input_channels=input_channels,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        tcn_channels.append(num_classes_pitch + 1)
        self.tcn = TemporalConvNet(
            input_size=conv_channels[-1] * input_features, 
            num_channels=tcn_channels,
            kernel_size=tcn_kernel,
            dropout=dropout,
        )

    def forward(self, x, tatum_frames):
            
        if x.shape[-2] != self.input_features:
            raise ValueError(f"Number of input features not match! expected{self.input_features} but got {x.shape[-2]}")
        # if tatum_frames.shape[-1] != self.num_tatums + 1:
        #     raise ValueError(f"Length of tatum_frames not match! expected{self.num_tatums + 1} but got {tatum_frames.shape[-1]}")
        
        tatum_frames = tatum_frames.unsqueeze(-2)
        # x: (batch_size, num_channels, num_features, length)
        # tatum_frames: (batch_size, num_channels=1, num_tatums + 1)

        x = self.cnn(x)
        # x: (batch_size, num_channels=conv_channels, num_features, length)
        old_shape = x.shape
        x = x.reshape(old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        # x: (batch_size, channel * num_features, length)
        # Pooling within tatums:
        x = segment_csr(x, tatum_frames, reduce="max")
        # x: (batch_size, num_channels=conv_channels * num_features, num_tatums)

        output = self.tcn(x).transpose(-1, -2)
        # output: (batch_size, num_tatums, num_classes + 1)
      
        return output[:, :, :-1], output[:, :, -1]

        # pitches_logits: (batch_size, num_tatums, num_pitches)
        # onsets_logits: (batch_size, num_tatums)