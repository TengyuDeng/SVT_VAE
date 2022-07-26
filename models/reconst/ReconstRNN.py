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

class RNN(nn.Module):

    def __init__(
        self,
        num_features,
        num_lstms=3,
        lstm_channels=512,
        dropout=0.,
        **args
        ):
        super().__init__()

        self.num_features = num_features
        self.lstm_channels = lstm_channels

        self.rnn = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else num_features + 1, 
            hidden_size=lstm_channels,
            dropout=dropout if i < num_lstms - 1 else 0.,
            normalize=True if i > 0 else False,
            )
            for i in range(num_lstms)
            ]
            )
        self.relu = torch.nn.ReLU()
        self.dense = nn.Linear(in_features=lstm_channels, out_features=output_features)

    def forward(self, pitch_rendered, onset, tatum_frames):
        
        # Input:
        #   pitch: (batch_size, num_tatums, num_features)
        #   onset: (batch_size, num_tatums)
        # Return:
        #   reconst_output: (batch_size, num_frames, num_features)
        pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
        onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
        reconst_input = torch.cat([pitch_rendered, onset], dim=-1)
        # reconst_input: (batch_size, num_frames, num_features + 1)
        reconst_input = self.rnn(reconst_input.transpose(0, 1)).transpose(0, 1)
        reconst_input = self.relu(reconst_input[..., :self.lstm_channels] + reconst_input[..., self.lstm_channels:])
        reconst_output = self.dense(reconst_input)
        
        return reconst_output

class RNNVariational(nn.Module):

    def __init__(
        self,
        output_features,
        z_features,
        phoneme_features=72,
        num_lstms=3,
        lstm_channels=512,
        dropout=0.,
        features=["pitch", "onset"],
        **args
        ):

        super().__init__()

        self.output_features = output_features
        self.features = features
        self.lstm_channels = lstm_channels
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

        self.rnn = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else input_features + z_features, 
            hidden_size=lstm_channels,
            dropout=dropout,
            normalize=True if i > 0 else False,
            )
            for i in range(num_lstms)
            ]
            )
        self.relu = torch.nn.ReLU()
        self.dense = nn.Linear(in_features=lstm_channels, out_features=output_features)
        
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
        reconst_input = self.rnn(reconst_input.transpose(0, 1)).transpose(0, 1)
        reconst_input = self.relu(reconst_input[..., :self.lstm_channels] + reconst_input[..., self.lstm_channels:])
        reconst_output = self.dense(reconst_input)

        return reconst_output

class RNNVariationalNew(nn.Module):
    """
    Reconstruct w/o z, and add the reconstructed results with weights
    """

    def __init__(
        self,
        output_features,
        z_features,
        phoneme_features=72,
        num_lstms=3,
        lstm_channels=512,
        dropout=0.,
        features=["pitch", "onset"],
        filter_alpha=0.,
        **args
        ):

        super().__init__()

        self.output_features = output_features
        self.features = features + ["z"]
        self.lstm_channels = lstm_channels
        self.filter_alpha = filter_alpha
        num_features = {
        "pitch": output_features,
        "onset": 1,
        "lyrics": phoneme_features,
        "z": z_features,
        }
        input_features = 0
        self.rnns = {}
        for feature_name in self.features:
            try:
                if feature_name != "onset":
                    self.rnns[feature_name] = RNNLayer(
                        input_size=num_features[feature_name], 
                        hidden_size=lstm_channels,
                        dropout=dropout,
                        normalize=False,
                    )
                    input_features += lstm_channels * 2
                else:
                    input_features += num_features[feature_name]
            except KeyError:
                raise RuntimeError(f"In constructing reconst model. No such feature {feature_name}.")
        self.rnns = nn.ModuleDict(self.rnns)
        self.rnn_after = RNNLayer(
            input_size=input_features,
            hidden_size=lstm_channels,
            dropout=dropout,
            normalize=True,
            )
        self.rnn_after_no_z = RNNLayer(
            input_size=input_features - lstm_channels * 2,
            hidden_size=lstm_channels,
            dropout=dropout,
            normalize=True,
            )
        self.relu = torch.nn.ReLU()
        self.dense = nn.Linear(in_features=lstm_channels, out_features=output_features)
        
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
            pitch_rendered = self.rnns['pitch'](pitch_rendered.transpose(0, 1))
            inputs.append(pitch_rendered)
        if "onset" in self.features:
            onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
            inputs.append(onset.transpose(0, 1))
        if "lyrics" in self.features:
            lyrics = lyrics.repeat_interleave(lyrics_downsample, dim=1)
            lyrics = adjust_shape(lyrics, ref_shape=(z.shape[-2], lyrics.shape[-1]))
            lyrics = self.rnns['lyrics'](lyrics.transpose(0, 1))
            inputs.append(lyrics)
        reconst_input_no_z = torch.cat(inputs, dim=-1)
        reconst_input_no_z = self.rnn_after_no_z(reconst_input_no_z).transpose(0, 1)
        reconst_input_no_z = self.relu(reconst_input_no_z[..., :self.lstm_channels] + reconst_input_no_z[..., self.lstm_channels:])
        reconst_output_no_z = self.dense(reconst_input_no_z)

        z = self.rnns['z'](z.transpose(0, 1))
        inputs.append(z)

        reconst_input = torch.cat(inputs, dim=-1)
        # reconst_input: (num_frames, batch_size, input_features + z_features)
        reconst_input = self.rnn_after(reconst_input).transpose(0, 1)
        reconst_input = self.relu(reconst_input[..., :self.lstm_channels] + reconst_input[..., self.lstm_channels:])
        reconst_output = self.dense(reconst_input)

        return reconst_output, reconst_output_no_z

# class RNNVariationalNewFilterV1(nn.Module):

#     def __init__(
#         self,
#         output_features,
#         z_features,
#         phoneme_features=72,
#         num_lstms=3,
#         lstm_channels=512,
#         dropout=0.,
#         features=["pitch", "onset"],
#         filter_alpha=0.,
#         **args
#         ):

#         super().__init__()

#         self.output_features = output_features
#         self.features = features + ["z"]
#         self.lstm_channels = lstm_channels
#         self.filter_alpha = filter_alpha
#         num_features = {
#         "pitch": output_features,
#         "onset": 1,
#         "lyrics": phoneme_features,
#         "z": z_features,
#         }
#         input_features = 0
#         self.rnns = {}
#         for feature_name in self.features:
#             try:
#                 if feature_name != "onset":
#                     self.rnns[feature_name] = RNNLayer(
#                         input_size=num_features[feature_name], 
#                         hidden_size=lstm_channels,
#                         dropout=dropout,
#                         normalize=False,
#                     )
#                     input_features += lstm_channels * 2
#                 else:
#                     input_features += num_features[feature_name]
#             except KeyError:
#                 raise RuntimeError(f"In constructing reconst model. No such feature {feature_name}.")
#         self.rnns = nn.ModuleDict(self.rnns)
#         self.rnn_after = RNNLayer(
#             input_size=input_features,
#             hidden_size=lstm_channels,
#             dropout=dropout,
#             normalize=True,
#             )
#         self.rnn_after_no_z = RNNLayer(
#             input_size=input_features - lstm_channels * 2,
#             hidden_size=lstm_channels,
#             dropout=dropout,
#             normalize=True,
#             )
#         self.relu = torch.nn.ReLU()
#         self.dense = nn.Linear(in_features=lstm_channels, out_features=output_features)
        
#     def forward(self, z, pitch_rendered, onset, tatum_frames, lyrics=None, lyrics_downsample=2):
        
#         # Input:
#         #   z: (batch_size, num_frames, z_features)
#         #   pitch: (batch_size, num_tatums, num_features)
#         #   onset: (batch_size, num_tatums)
#         #   lyrics: (batch_size, num_frames, phoneme_features)
#         # Return:
#         #   reconst_output: (batch_size, num_frames, num_features)
#         inputs = []
#         if "pitch" in self.features:
#             pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
#             pitch_rendered = self.rnns['pitch'](pitch_rendered.transpose(0, 1))
#             inputs.append(pitch_rendered)
#         if "onset" in self.features:
#             onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
#             inputs.append(onset.transpose(0, 1))
#         if "lyrics" in self.features:
#             lyrics = lyrics.repeat_interleave(lyrics_downsample, dim=1)
#             lyrics = adjust_shape(lyrics, ref_shape=(z.shape[-2], lyrics.shape[-1]))
#             lyrics = self.rnns['lyrics'](lyrics.transpose(0, 1))
#             inputs.append(lyrics)
#         reconst_input_no_z = torch.cat(inputs, dim=-1)
#         reconst_input_no_z = self.rnn_after_no_z(reconst_input_no_z).transpose(0, 1)
#         reconst_input_no_z = self.relu(reconst_input_no_z[..., :self.lstm_channels] + reconst_input_no_z[..., self.lstm_channels:])
#         reconst_output_no_z = self.dense(reconst_input_no_z)

#         z = self.rnns['z'](z.transpose(0, 1))
#         inputs.append(z)

#         reconst_input = torch.cat(inputs, dim=-1)
#         # reconst_input: (num_frames, batch_size, input_features + z_features)
#         reconst_input = self.rnn_after(reconst_input).transpose(0, 1)
#         reconst_input = self.relu(reconst_input[..., :self.lstm_channels] + reconst_input[..., self.lstm_channels:])
#         reconst_output_filter = self.dense(reconst_input)
#         reconst_output = reconst_output_filter * reconst_output_no_z

#         return reconst_output, reconst_output_no_z

# class RNNVariationalNewFilterV2(nn.Module):

#     def __init__(
#         self,
#         output_features,
#         z_features,
#         phoneme_features=72,
#         num_lstms=3,
#         lstm_channels=512,
#         dropout=0.,
#         features=["pitch", "onset"],
#         filter_alpha=0.,
#         **args
#         ):

#         super().__init__()

#         self.output_features = output_features
#         self.features = features
#         self.lstm_channels = lstm_channels
#         self.filter_alpha = filter_alpha
#         num_features = {
#         "pitch": output_features,
#         "onset": 1,
#         "lyrics": phoneme_features,
#         "z": z_features,
#         }
#         input_features = 0
#         self.rnns = {}
#         for feature_name in self.features:
#             try:
#                 if feature_name != "onset":
#                     self.rnns[feature_name] = RNNLayer(
#                         input_size=num_features[feature_name], 
#                         hidden_size=lstm_channels,
#                         dropout=dropout,
#                         normalize=False,
#                     )
#                     input_features += lstm_channels * 2
#                 else:
#                     input_features += num_features[feature_name]
#             except KeyError:
#                 raise RuntimeError(f"In constructing reconst model. No such feature {feature_name}.")
#         self.rnns = nn.ModuleDict(self.rnns)
#         self.rnn_z = nn.Sequential(
#             RNNLayer(
#                 input_size=z_features, 
#                 hidden_size=lstm_channels,
#                 dropout=dropout,
#                 normalize=False,
#             ),
#             RNNLayer(
#                 input_size=lstm_channels * 2,
#                 hidden_size=lstm_channels,
#                 dropout=dropout,
#                 normalize=True,
#             ),
#         )
        
#         self.rnn_after_no_z = RNNLayer(
#             input_size=input_features,
#             hidden_size=lstm_channels,
#             dropout=dropout,
#             normalize=True,
#             )
#         self.relu = torch.nn.ReLU()
#         self.dense = nn.Linear(in_features=lstm_channels, out_features=output_features)
        
#     def forward(self, z, pitch_rendered, onset, tatum_frames, lyrics=None, lyrics_downsample=2):
        
#         # Input:
#         #   z: (batch_size, num_frames, z_features)
#         #   pitch: (batch_size, num_tatums, num_features)
#         #   onset: (batch_size, num_tatums)
#         #   lyrics: (batch_size, num_frames, phoneme_features)
#         # Return:
#         #   reconst_output: (batch_size, num_frames, num_features)
#         inputs = []
#         if "pitch" in self.features:
#             pitch_rendered = repeat_with_tatums(pitch_rendered, tatum_frames)
#             pitch_rendered = self.rnns['pitch'](pitch_rendered.transpose(0, 1))
#             inputs.append(pitch_rendered)
#         if "onset" in self.features:
#             onset = repeat_with_tatums(onset[:,:,None], tatum_frames)
#             inputs.append(onset.transpose(0, 1))
#         if "lyrics" in self.features:
#             lyrics = lyrics.repeat_interleave(lyrics_downsample, dim=1)
#             lyrics = adjust_shape(lyrics, ref_shape=(z.shape[-2], lyrics.shape[-1]))
#             lyrics = self.rnns['lyrics'](lyrics.transpose(0, 1))
#             inputs.append(lyrics)
#         reconst_input_no_z = torch.cat(inputs, dim=-1)
#         reconst_input_no_z = self.rnn_after_no_z(reconst_input_no_z).transpose(0, 1)
#         reconst_input_no_z = self.relu(reconst_input_no_z[..., :self.lstm_channels] + reconst_input_no_z[..., self.lstm_channels:])
#         reconst_output_no_z = self.dense(reconst_input_no_z)

#         z = self.rnn_z(z.transpose(0, 1)).transpose(0, 1)
#         z = self.relu(z[..., :self.lstm_channels] + z[..., self.lstm_channels:])
#         reconst_output_filter = self.dense(z)
#         reconst_output = reconst_output_filter * reconst_output_no_z

#         return reconst_output, reconst_output_no_z
