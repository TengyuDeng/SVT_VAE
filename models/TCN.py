import sys
sys.path.append("..")
from utils import get_padding

from torch import nn
from torch_scatter import segment_csr
from .layers import *


class TCN(nn.Module):

    def __init__(
        self,
        num_classes_pitch,
        input_features=80,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 1],
        kernel_sizes=[5, 5, 3, 3, 3, 1],
        dropout=0.,
        lstm_channels=512,
        **args
        ):

        super(TCN, self).__init__()


    def forward(self, x, tatum_frames):
        pass