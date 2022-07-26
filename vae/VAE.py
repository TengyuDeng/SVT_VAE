import torch, numpy as np
from .utils import *

class VAE:
    def __init__(self, z_est_model, x_reconst_model, language_model, num_sample=1, loss_weights=None):
        """ 
        z_est_model:
          Input: 
            x_input: (batch_size, num_channels, num_features, num_frames)
            tatum_frames: (batch_size, num_tatums + 1)
          Output:
            q_z_mean_x: (batch_size, num_frames, dim_z)
            q_z_std_x: (batch_size, num_frames, dim_z)

        x_reconst_model:
          Input:
            z: (batch_size, num_frames, dim_z)
            pitch: (batch_size, num_tatums, num_pitches)
            onset: (batch_size, num_tatums)
            tatum_frames: (batch_size, num_tatums + 1)
          Output:
            p_x_mean_zpn: (batch_size, num_frames, num_features)
            p_x_std_zpn: (batch_size, num_frames, num_features)

        language_model:
          See language_model module.
        """
        self.z_est_model = z_est_model
        self.x_reconst_model = x_reconst_model
        self.language_model = language_model
        self.num_sample = num_sample
        self.loss_weights = loss_weights

    def __call__(
        self,
        pitch, onset,
        x_target,
        input_lengths,
        tatum_frames,
        lyrics=None,
        lyrics_downsample=2,
        supervised=False,
        reconst_input=None,
        ):
        """
        Input:
          x_target: (batch_size, num_features, num_frames)
          input_lengths: (batch_size)
          tatum_frames: (batch_size, num_tatumes)mse_loss
          pitch: (batch_size, num_tatums, num_pitches)
          onset: (batch_size, num_tatums)

        Output:
          loss: torch.Tensor
        """

        # p(Z): prior distribution of Z
        # q(Z|X): posterior distribution of Z
        if reconst_input is None:
            reconst_input = x_target
        q_z_mean_x, q_z_std_x = self.z_est_model(reconst_input, tatum_frames)
        batch_size, num_tatums, dim_z = q_z_mean_x.shape
        z_dist_post = torch.distributions.normal.Normal(
            loc=q_z_mean_x, scale=q_z_std_x
            )
                
        # Reconstruction Error
        reconst_loss = []
        for i in range(self.num_sample):
            z_sample = z_dist_post.rsample()
            reconst = self.x_reconst_model(z_sample, pitch, onset, tatum_frames, lyrics=lyrics, lyrics_downsample=lyrics_downsample)
            if isinstance(reconst, tuple):
                reconst_with_z, reconst_no_z = reconst
                # print(f"""
                #     current_loss = (
                #     {self.loss_weights['with_z']} * {mse_loss(reconst_with_z.transpose(-1, -2), x_target, input_lengths)} + 
                #     {self.loss_weights['no_z']} * {mse_loss(reconst_no_z.transpose(-1, -2), x_target, input_lengths)}
                #     )
                #     """)
                current_loss = (
                    self.loss_weights['with_z'] * mse_loss(reconst_with_z.transpose(-1, -2), x_target, input_lengths) + 
                    self.loss_weights['no_z'] * mse_loss(reconst_no_z.transpose(-1, -2), x_target, input_lengths)
                    )
            else:
                current_loss = mse_loss(reconst.transpose(-1, -2), x_target, input_lengths)
            reconst_loss.append(current_loss)

        reconst_loss = torch.mean(torch.stack(reconst_loss))
        
        # KL divergence between posterior and prior distributions of Z
        kl_loss = []
        for j in range(batch_size):
            z_dist_prior = torch.distributions.normal.Normal(
                loc = torch.zeros_like(q_z_mean_x[j, :input_lengths[j], :]),
                scale = torch.ones_like(q_z_std_x[j, :input_lengths[j], :])
                )
            z_dist_post = torch.distributions.normal.Normal(
                loc=q_z_mean_x[j, :input_lengths[j], :], scale=q_z_std_x[j, :input_lengths[j], :]
                )
            kl_loss.append(torch.mean(torch.distributions.kl.kl_divergence(z_dist_post, z_dist_prior)))
        kl_loss = torch.mean(torch.stack(kl_loss))
        return reconst_loss + kl_loss

    # Language Model (not implemented)
    # if supervised:        
    #         # Prior log probs of pitch and onset
    #         log_prob_pitch, log_prob_onset = self.language_model.log_prob(pitch, onset)
    #         log_prob_pitch = torch.mean(log_prob_pitch)
    #         log_prob_onset = torch.mean(log_prob_onset)
             
    #         return reconst_loss + kl_loss - log_prob_pitch - log_prob_onset
    
    #     else:

    #         # Expected prior log probs w.r.t posterior distribution for P, N
    #         exp_log_prob_pn = self.language_model.expected_log_prob(pitch_logits, onset_logits)
    #         exp_log_prob_pn = torch.mean(exp_log_prob_pn)

    #         # Entropy of posterior distribution of P, N
    #         ent_pitch = torch.mean(ent_categorical(pitch_logits, dim=-1))
    #         ent_onset = torch.mean(ent_bernoulli(onset_logits))
    #         return reconst_loss - exp_log_prob_pn + kl_loss - ent_pitch - ent_onset