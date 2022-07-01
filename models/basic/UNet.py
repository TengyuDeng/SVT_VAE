from torch import nn
import torch

# def get_padding(kernel_size):
#     if isinstance(kernel_size, int):
#         padding = kernel_size // 2
#     else:
#         padding = tuple(x // 2 for x in kernel_size)
#     return padding

def adjust_shape(x, ref):
    """
    Inputs: 
      x: (batch_size, channel_x, H_x, W_x)
      ref: (batch_size, channel_ref, H_ref, W_ref)
    Returns:
      x_adjusted: (batch_size, channel_ref, H_ref, W_ref)
    Only appliable when 
    (H_x - H_ref) * (W_x - W_ref) >= 0
    """
    _, _, H_x, W_x = x.shape
    _, _, H_ref, W_ref = ref.shape
    if (H_x - H_ref) * (W_x - W_ref) < 0:
        raise ValueError(f"Wrong shapes: {x.shape}, {ref.shape}")
    
    if H_x >= H_ref:
        return x[:, :, :H_ref, :W_ref]
    else:
        # padding = (W_ref - W_x, 0, H_ref - H_x, 0)
        padding = (0, W_ref - W_x, 0, H_ref - H_x)
        return nn.ZeroPad2d(padding)(x)

class DownSampleLayer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=2, dropout=0.2):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.dropout(x)

class UpSampleLayer(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=2, dropout=0.2):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.deconv = nn.ConvTranspose2d(output_channels, output_channels//2, stride, stride=stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.deconv(self.dropout(x))

class UNet(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=4,
        stride=2,
        kernel_sizes=None,
        dropout=0.,
        ):

        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers
        if len(kernel_sizes) != num_layers:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        
        self.downsampling = nn.Sequential(*[
            DownSampleLayer(
                input_channels * 2 ** i,
                input_channels * 2 ** (i + 1),
                kernel_sizes[i],
                stride=stride, dropout=dropout,
                ) for i in range(num_layers)
            ])

        self.upsampling = nn.Sequential(*[
            UpSampleLayer(
                input_channels * 2 ** num_layers if i == 0 else input_channels * 2 ** (num_layers - i + 1),
                input_channels * 2 ** (num_layers - i),
                kernel_sizes[num_layers - i - 1],
                stride=stride, dropout=dropout,
                ) for i in range(num_layers)
            ])
        
        self.fc = nn.Sequential(
            nn.Linear(input_channels, output_channels),
            nn.ReLU(),
            )
    def forward(self, x):

        inter_xs = [x]
        for i in range(self.num_layers):
            x = self.downsampling[i](x)
            inter_xs.append(x)

        for i in range(self.num_layers):
            inter_x = inter_xs.pop()
            if i > 0:
                x = adjust_shape(x, inter_x)
                x_input = torch.cat([x, inter_x], dim=1)
            else:
                x_input = x
            x = self.upsampling[i](x_input)
        
        output = self.fc(x.transpose(1, -1)).transpose(1, -1)
        return output