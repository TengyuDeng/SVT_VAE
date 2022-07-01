import torch
from torch import nn
from torch_scatter import segment_csr
from ..basic.UNet import UNet
from ..basic.RNN import RNNLayer

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
    
class ReconstUNet(nn.Module):

    def __init__(
        self,
        unet_layers=4,
        kernel_sizes=None,
        dropout=0.,
        **args
        ):

        super().__init__()
        self.unet = UNet(
            2,
            1,
            num_layers=unet_layers,
            stride=2,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            )


    def forward(self, pitch_rendered, onset, tatum_frames):
        
        # Input:
        #   pitch: (batch_size, num_tatums, num_features)
        #   onset: (batch_size, num_tatums)
        # Return:
        #   reconst_output: (batch_size, num_frames, num_features)
        num_features = pitch_rendered.shape[-1]
        pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
        onset = repeat_with_tatums(onset[:,:,None].repeat([1, 1, num_features]), tatum_frames)
        reconst_input = torch.stack([pitch_rendered, onset], dim=1)
        # reconst_input: (batch_size, 2, num_frames, num_features)
        reconst_output = self.unet(reconst_input).squeeze()
        
        return reconst_output

class ReconstUNetPitchOnly(nn.Module):

    def __init__(
        self,
        unet_layers=4,
        kernel_sizes=None,
        dropout=0.,
        **args
        ):

        super().__init__()
        self.unet = UNet(
            1,
            1,
            num_layers=unet_layers,
            stride=2,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            )


    def forward(self, pitch_rendered, onset, tatum_frames):
        
        # Input:
        #   pitch: (batch_size, num_tatums, num_features)
        #   onset: (batch_size, num_tatums)
        # Return:
        #   reconst_output: (batch_size, num_frames, num_features)
        num_features = pitch_rendered.shape[-1]
        pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
        reconst_input = pitch_rendered.unsqueeze(1)
        # # reconst_input: (batch_size, num_frames, num_features, 2)
        # reconst_input: (batch_size, 2, num_frames, num_features)
        reconst_output = self.unet(reconst_input).squeeze()
        
        return reconst_output