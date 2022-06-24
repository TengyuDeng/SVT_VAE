import sys
sys.path.append("..")
from utils import downsample_length, get_padding
import os
from torch import nn
        
class RNNLayer(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.2):

        super(RNNLayer, self).__init__()

        self.norm = nn.LayerNorm(input_size)
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x, (h_n, c_n) = self.bilstm(self.norm(x))
        return self.dropout(x)
