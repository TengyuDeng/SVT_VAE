import torch
from torch import nn

class Vanila(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(()))

    def expected_log_prob(self, pitch_logits, onset_logits):
        """
        Input:
          pitch_logits: (batch_size, num_tatums, num_pitches)
          onset_logits: (batch_size, num_tatums)
        """
        return 0 * self.param

    def log_prob(self, pitch, onset):
        return 0 * self.param, 0 * self.param