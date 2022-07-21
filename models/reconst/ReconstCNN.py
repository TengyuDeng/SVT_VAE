import torch
from torch import nn
from torch_scatter import segment_csr
from ..basic.CNN import ConvNN
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

def adjust_shape(x, ref_shape):
    """
    Inputs: 
      x: (batch_size, channel_x, H_x, W_x)
      ref: (batch_size, channel_ref, H_ref, W_ref)
    Returns:
      x_adjusted: (batch_size, channel_ref, H_ref, W_ref)
    Only appliable when 
    (H_x - H_ref) * (W_x - W_ref) >= 0
    """
    H_x, W_x = x.shape[-2:]
    H_ref, W_ref = ref_shape
    if (H_x - H_ref) * (W_x - W_ref) < 0:
        raise ValueError(f"Wrong shapes: {x.shape}, {ref_shape}")
    
    if H_x >= H_ref:
        return x[..., :H_ref, :W_ref]
    else:
        padding = (0, W_ref - W_x, 0, H_ref - H_x)
        return nn.ZeroPad2d(padding)(x)


class CNN(nn.Module):

    def __init__(
        self,
        num_convs=4,
        conv_channels=[32,32,32,1],
        kernel_sizes=[3,3,3,3],
        dropout=0.,
        **args
        ):

        super().__init__()
        if conv_channels[-1] != 1:
            raise ValueError(f"the output must be 1 channel, but got {conv_channels[-1]}")
        self.cnn = ConvNN(
            input_channels=2,
            num_convs=num_convs,
            conv_channels=conv_channels,
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
        reconst_output = self.cnn(reconst_input).squeeze()
        return reconst_output

class LSTMCNN(nn.Module):

    def __init__(
        self,
        num_features,
        num_convs=4,
        conv_channels=[32,32,32,1],
        kernel_sizes=[3,3,3,3],
        dropout=0.,
        **args
        ):
        super().__init__()
        if conv_channels[-1] != 1:
            raise ValueError(f"the output must be 1 channel, but got {conv_channels[-1]}")
        self.num_features = num_features
        self.lstm = RNNLayer(
            num_features + 1,
            num_features,
            dropout=dropout,
            )
        self.cnn = ConvNN(
            input_channels=1,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

    def forward(self, pitch_rendered, onset, tatum_frames):
        
        # Input:
        #   pitch: (batch_size, num_tatums, num_features)
        #   onset: (batch_size, num_tatums)
        # Return:
        #   reconst_output: (batch_size, num_frames, num_features)
        pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
        # onset= repeat_with_tatums(onset[:,:,None].repeat([1, 1, num_features]), tatum_frames)
        # reconst_input = torch.stack([pitch_rendered, onset], dim=-1)
        # # reconst_input: (batch_size, num_frames, num_features, 2)
        onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
        reconst_input = torch.cat([pitch_rendered, onset], dim=-1)
        # reconst_input: (batch_size, num_frames, num_features + 1)
        reconst_input = self.lstm(reconst_input.transpose(0, 1)).transpose(0, 1).unsqueeze(1)
        reconst_input = reconst_input[..., :self.num_features] + reconst_input[..., self.num_features:]
        reconst_output = self.cnn(reconst_input).squeeze()
        
        return reconst_output

class LSTMCNNVariational(nn.Module):

    def __init__(
        self,
        output_features,
        z_features,
        phoneme_features=72,
        num_convs=4,
        conv_channels=[32,32,32,1],
        kernel_sizes=[3,3,3,3],
        dropout=0.,
        features=["pitch", "onset"],
        **args
        ):

        super().__init__()
        if conv_channels[-1] != 1:
            raise ValueError(f"the output must be 1 channel, but got {conv_channels[-1]}")
        self.output_features = output_features
        self.features = features
        num_features = {
        "pitch": output_features,
        "onset": 1,
        "lyrics": phoneme_features,
        }
        input_features = 0
        for feature_name in features:
            try:
                input_features += num_features[feature_name]
            except KeyError:
                raise RuntimeError(f"In constructing reconst model. No such feature {feature_name}.")
        self.lstm = RNNLayer(
            input_features + z_features,
            output_features,
            dropout=dropout,
            )
        self.cnn = ConvNN(
            input_channels=1,
            num_convs=num_convs,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

    def forward(self, z, pitch_rendered, onset, tatum_frames, lyrics=None, lyrics_downsample=2):
        
        # Input:
        #   z: (batch_size, num_frames, z_features)
        #   pitch: (batch_size, num_tatums, num_features)
        #   onset: (batch_size, num_tatums)
        #   lyrics: (batch_size, num_frames, phoneme_features)
        # Return:
        #   reconst_output: (batch_size, num_frames, num_features)
        inputs = []
        if "pitch" in self.features:
            pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
            inputs.append(pitch_rendered)
        if "onset" in self.features:
            onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
            inputs.append(onset)
        if "lyrics" in self.features:
            lyrics = lyrics.repeat_interleave(lyrics_downsample, dim=1)
            lyrics = adjust_shape(lyrics, ref_shape=(z.shape[-2], lyrics.shape[-1]))
            inputs.append(lyrics)
        inputs.append(z)
        reconst_input = torch.cat(inputs, dim=-1)
        # reconst_input: (batch_size, num_frames, input_features + z_features)
        reconst_input = self.lstm(reconst_input.transpose(0, 1)).transpose(0, 1).unsqueeze(1)
        reconst_input = reconst_input[..., :self.output_features] + reconst_input[..., self.output_features:]
        reconst_output = self.cnn(reconst_input).squeeze()

        return reconst_output
