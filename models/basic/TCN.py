from torch import nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size].contiguous()

class TCNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation,
            ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
            )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        tcn_layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcn_layers.append(TCNLayer(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                dropout=dropout,
                ))

        self.network = nn.Sequential(*tcn_layers)

    def forward(self, x):
        return self.network(x)