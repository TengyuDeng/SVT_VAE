import torch, torchaudio
from torch import nn
import numpy as np, librosa

class Renderer(nn.Module):

    def __init__(
        self,
        num_harmony=20,
        frs=None,
        feature_type="stft",
        sr=22050,
        n_fft=2048,
        n_mels=128,
        n_bins=84,
        bins_per_octave=12,
        fmin=None,
        fmax=None,
        init_sigma=10.,
        init_omega=1.,
        ):
        """
        Parameters:
          frs:  The frequencies of each bin in the rendered features.

          feature_type and corresponding parameters only works if frs is not specified.
          feature_type:
            stft, mel, or cqt
        """
        super().__init__()
        if frs is None:
            if feature_type == "stft":
                frs = torch.arange(1 + n_fft // 2) / n_fft * sr
            elif feature_type == "cqt":
                if fmin is None:
                    fmin = 32.70
                frs = fmin * torch.pow(2, torch.arange(n_bins) / bins_per_octave)
            elif feature_type == "mel":
                if fmin is None:
                    fmin = 0.
                if fmax is None:
                    fmax = sr / 2
                mel_min, mel_max = librosa.hz_to_mel([fmin, fmax])
                frs = librosa.mel_to_hz(mel_min + np.arange(n_mels) * (mel_max - mel_min) / (n_mels - 1))
                frs = torch.tensor(frs, dtype=torch.float32)
        frs_input = 440. * torch.pow(2, (torch.arange(128) - 69) / 12)
        
        self.fmin = fmin
        self.fmax = fmax
        num_f_bins = len(frs)
        num_in_bins = len(frs_input)
        frs = frs[:, None, None].repeat([1, num_in_bins, num_harmony])
        frs_input = frs_input[None, :, None].repeat([num_f_bins, 1, num_harmony])

        # self.sigmas = nn.Parameter(torch.full((num_harmony,), init_sigma))
        # self.omegas = nn.Parameter(torch.full((num_harmony,), init_omega))
        self.sigmas = init_sigma * torch.ones(num_harmony)
        self.omegas = init_omega * torch.ones(num_harmony)
        self.r_filter = self.omegas * torch.exp(- (frs - torch.arange(1, 1 + num_harmony) * frs_input) ** 2 / (2 * self.sigmas ** 2))
        self.r_filter = torch.sum(self.r_filter, dim=-1)
        # if feature_type == "mel":
        #     if fmin is None:
        #         fmin = 0.
        #     self.post_filter = torchaudio.transforms.MelScale(
        #         n_mels=n_mels, sample_rate=sr, n_stft=1 + n_fft // 2, f_min=fmin, f_max=fmax
        #         )
        # else:
        #     self.post_filter = nn.Identity()

    def forward(self, pitches):
        # pitches: (..., num_frames, num_pitches=129)
        pitches = pitches[..., :128].transpose(-1, -2)
        # pitches: (..., num_pitches=128, num_frames)
        rendered = torch.matmul(self.r_filter, pitches)
        # rendered = self.post_filter(torch.matmul(self.r_filter, pitches))
        return rendered.transpose(-1, -2)
        # rendered: (..., num_frames, len(frs) or n_mels)

    def to(self, *args, **kwargs):
        self.r_filter = self.r_filter.to(*args, **kwargs)
        # self.post_filter = self.post_filter.to(*args, **kwargs)
        return self