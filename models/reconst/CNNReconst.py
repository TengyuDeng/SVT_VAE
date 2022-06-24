import torch
from torch import nn
from torch_scatter import segment_csr
from ..basic.CNN import ConvNN

ALPHA = 1e-4

def repeat_with_tatums(features, tatum_frames):
    """
    Input:
      features: (batch_size, num_tatums, num_features)
      tatum_frames: (batch_size, num_tatums + 1)
    Return:
      features_repeat: (batch_size, max(tatum_frames[:, -1]), num_features)
    """
    batch_size = features.shape[0]
    new_features = [features[i, ...].repeat_interleave(torch.diff(tatum_frames[i]), dim=0) for i in range(batch_size)]
    return torch.nn.utils.rnn.pad_sequence(new_features, batch_first=True)
    
class CNNReconst(nn.Module):

    def __init__(
        self,
        input_features,
        output_features,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 2],
        kernel_sizes=[5, 5, 3, 3, 3, 1],
        dropout=0.,
        **args
        ):

        super().__init__()
        if conv_channels[-1] != 2:
            raise ValueError(f"the output must be 2 channels (mean, std), but got {conv_channels[-1]}")
        
        self.cnn = ConvNN(
            input_channels=3,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.fc = nn.Linear(input_features, output_features)

    def forward(self, z, pitch, onset, tatum_frames):
        
        batch_size, num_frames, dim_z = z.shape
        if pitch.shape[-1] != dim_z + 1:
            raise ValueError("z and pitch (without rest) must be concatenatable!")
        # Is is good to drop rest note?
        pitch = repeat_with_tatums(pitch[..., :-1], tatum_frames)
        onset = repeat_with_tatums(onset[:,:,None].repeat([1, 1, dim_z]), tatum_frames)
        
        features = torch.stack([z, pitch, onset], dim=1)
        # features: (batch_size, channels = 3, num_frames, dim_z)
        output = self.fc(self.cnn(features))
        # output: (batch_size, num_channels=2, num_features, output_frames)
        
        return output[:, 0, ...], torch.abs(output[:, 1, ...]) + ALPHA
        # reconst_mean: (batch_size, num_frames, output_features)
        # reconst_std: (batch_size, num_frames, output_features)
