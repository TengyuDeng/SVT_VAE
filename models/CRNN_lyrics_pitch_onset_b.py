import sys
sys.path.append("..")
from utils import downsample_length, get_padding, get_filter_midi_to_mel
import torch
from torch import nn
from torch_scatter import segment_csr
from .layers import *

class CRNN_lyrics_pitch_onset_b(nn.Module):

    def __init__(
        self,
        num_classes_lyrics,
        num_classes_pitch,
        input_features=80,
        input_channels=1,
        num_convs=6,
        conv_channels=[64, 32, 32, 32, 32, 16],
        kernel_sizes=[5, 5, 3, 3, 3, 3],
        dropout=0.,
        num_lstms=3,
        lstm_channels=512,
        down_sample=2,
        **args
        ):

        super(CRNN_lyrics_pitch_onset_b, self).__init__()

        if len(conv_channels) != num_convs:
            raise ValueError(f"Expect conv_channels to have {num_convs} elements but got {len(conv_channels)}!")
        if len(kernel_sizes) != num_convs:
            raise ValueError(f"Expect kernel_sizes to have {num_convs} elements but got {len(kernel_sizes)}!")
        self.num_classes_lyrics = num_classes_lyrics
        self.num_classes_pitch = num_classes_pitch
        self.input_features = input_features
        self.midi_to_mel_filter = torch.tensor(get_filter_midi_to_mel(n_classes=num_classes_pitch, n_mels=input_features), dtype=torch.float)
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.down_sample = down_sample

        
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels + 2,
                out_channels=conv_channels[0], 
                kernel_size=kernel_sizes[0], 
                padding=get_padding(kernel_sizes[0]),
                ),
            *[
            ResCNNLayer(
                in_channels=conv_channels[i - 1], 
                out_channels=conv_channels[i], 
                kernel_size=kernel_sizes[i],
                dropout=dropout,
                )
            for i in range(1, num_convs)
            ],
            )
        
        self.pooling_lyrics = nn.MaxPool2d(
            kernel_size=3, 
            stride=self.down_sample, 
            padding=get_padding(3),
        )

        # self.downsampling_lyrics = ResCNNLayer(
        #     in_channels=conv_channels[-1],
        #     out_channels=conv_channels[-1],
        #     kernel_size=kernel_sizes[-1],
        #     stride=self.down_sample,
        #     dropout=dropout,
        # )

        self.rnn_lyrics = nn.Sequential(
            *[
            RNNLayer(
            input_size=lstm_channels * 2 if i > 0 else conv_channels[-1] * downsample_length(input_features, self.down_sample), 
            hidden_size=lstm_channels,
            )
            for i in range(num_lstms)
            ]
            )
        self.dense_lyrics = nn.Sequential(
            nn.Linear(in_features=lstm_channels * 2, out_features=lstm_channels),
            nn.ReLU(),
            nn.Linear(in_features=lstm_channels, out_features=num_classes_lyrics),
            )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.midi_to_mel_filter = self.midi_to_mel_filter.to(*args, **kwargs)
        return self

    def forward(self, x, pitches, onsets):
            
        if x.shape[-2] != self.input_features:
            raise ValueError(f"Number of input features not match! expected {self.input_features} but got {x.shape[-2]}")
        if pitches.shape[-2] != self.num_classes_pitch:
            raise ValueError(f"Length of pitches not match! expected {self.num_classes_pitch} but got {pitches.shape[-2]}")
        
        if self.input_channels == 1:
            if x.ndim == 3:
                x = x.unsqueeze(-3)
            # x: (batch_size, channel=1, input_features, length)
        else:
            if x.shape[1] == 1:
                x = x.repeat_interleave(self.input_channels, dim=1)
            if x.ndim == 3:
                x = x.unsqueeze(1).repeat_interleave(self.input_channels, dim=1)
            # x: (batch_size, channel>1, input_features, length)

        if onsets.ndim == 2:
            onsets = onsets.unsqueeze(1)
        # onsets: (batch_size, 1, length)
        # pitches: (batch_size, num_classes_pitch, length)
        onsets = onsets.repeat_interleave(self.input_features, dim=1)
        pitches = torch.matmul(self.midi_to_mel_filter, pitches)
        # onsets: (batch_size, input_features, length)
        # pitches: (batch_size, input_features, length)
        pitches_onsets = torch.stack([pitches, onsets], dim=1)

        x = torch.cat([x, pitches_onsets], dim=1)

        x = self.cnn(x)
        # x: (batch_size, channel=conv_channels, new_feature_num, new_length)
          
        x_lyrics = self.pooling_lyrics(x)

        # x_lyrics = self.downsampling_lyrics(x)
        # x_lyrics: (batch_size, channel, new_feature_num, new_length)
        old_shape = x_lyrics.shape
        x_lyrics = x_lyrics.reshape(old_shape[0], old_shape[1] * old_shape[2], old_shape[3])
        # x: (batch_size, channel * new_feature_num, new_length)
        x_lyrics = x_lyrics.permute(2, 0, 1)
        # x_lyrics: (new_length, batch_size, feature_num)

        x_lyrics = self.rnn_lyrics(x_lyrics)
        output_lyrics = self.dense_lyrics(x_lyrics)
        # x_lyrics: (new_length, batch_size, num_classes)

        return output_lyrics
        # output_pitch: (batch_size, num_classes + 1, num_tatums)
        # output_lyrics: (new_length, batch_size, num_classes)
