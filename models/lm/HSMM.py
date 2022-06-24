import torch
from torch import nn

class HSMM(nn.Module):
    def __init__(self, num_pitches=129, **args):
        super().__init__()
        self.init_probs = torch.nn.Parameter(torch.randn(()))

    def expected_log_prob(self, pitch_logits, onset_logits):
        """
        Input:
          pitch_logits: (batch_size, num_tatums, num_pitches)
          onset_logits: (batch_size, num_tatums)
        """
        pass

    def log_prob(self, pitch, onset):
        """
        Input:
          pitch: (batch_size, num_pitches, num_tatums)
          onset: (batch_size, num_tatums)
        """

        log_prob_pitch = 0.
        log_prob_onset = 0.

        return log_prob_pitch, log_prob_onset