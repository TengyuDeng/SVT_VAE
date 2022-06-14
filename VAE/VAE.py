import torch, numpy as np
from utils import *

class VAE:
    def __init__(self, z_est_model, p_n_est_model, x_reconst_model, language_model, num_sample=1, loss_weights=None):
        """ 
        z_est_model:
          Input: 
            x_input: (batch_size, num_channels, num_features, num_frames)
            tatum_frames: (batch_size, num_tatumes)
          Output:
            q_z_mean_x: (batch_size, num_tatums, dim_z)
            q_z_var_x: (batch_size, num_tatums, dim_z, dim_z)
        
        p_n_est_model:
          Input:
            x_input: (batch_size, num_channels, num_features, num_frames)
            tatum_frames: (batch_size, num_tatumes)
          Output:
            pitches_logits: (batch_size, num_tatums, num_pitches)
            onsets_logits: (batch_size, num_tatums)

        x_reconst_model:
          Input:
            z: (batch_size, num_tatums, dim_z)
            pitch: (batch_size, num_tatums, num_pitches)
            onset: (batch_size, num_tatums)
            tatum_frames: (batch_size, num_tatumes)
          Output:
            p_x_mean_zpn: (batch_size, num_frames, num_features)
            p_x_var_zpn: (batch_size, num_frames, num_features, num_features)

        language_model:
          See language_model module.
        """
        self.z_est_model = z_est_model
        self.p_n_est_model = p_n_est_model
        self.x_reconst_model = x_reconst_model
        self.language_model = language_model
        self.num_sample = num_sample
        self.pitch_loss_fn = torch.nn.CrossEntropyLoss(),
        self.onset_loss_fn = torch.nn.BCEWithLogitsLoss(),
        self.loss_weights = loss_weights

    def __call__(self,
        x_input,
        tatum_frames,
        x_target,
        supervised=False,
        pitch=None, 
        onset=None,
        ):
    """
    Input:
      x_input: (batch_size, num_channels, num_features, num_frames)
      tatum_frames: (batch_size, num_tatumes)
      x_target: (batch_size, num_features, num_frames)
      pitch: (batch_size, num_pitches, num_tatums)
      onset: (batch_size, num_tatums)
    """
    x_target = x_target.transpose(-1, -2)

    if supervised:
        if pitch is None or onset is None:
            raise ValueError("For supervised learning, pitch and onset labels must be given!")
        
        # p(Z): prior distribution of Z
        # q(Z|X): posterior distribution of Z
        q_z_mean_x, q_z_var_x = self.z_est_model(x_input, tatum_frames)
        batch_size, num_tatums, dim_z = q_z_mean_x.shape
        z_dist_prior = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = torch.zeros_like(q_z_mean_x), 
            covariance_matrix = torch.eye(dim_z)[None, None, :].repeate([batch_size, num_tatums, 1, 1])
            )
        z_dist_post = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=q_z_mean_x, covariance_matrix=q_z_var_x
            )

        # q(P, N|X): posterior distribution of P, N
        pitch_logits, onset_logits = self.p_n_est_model(x_input, tatum_frames)
        
        # Reconstruction Error
        reconst_loss = torch.tensor(0.)
        for i in range(self.num_sample):
            z_sample = z_dist_post.rsample()
            p_x_mean_zpn, p_x_var_zpn = self.x_reconst_model(z_sample, pitch, onset, tatum_frames)
            x_dist_reconst = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=p_x_mean_zpn, covariance_matrix=p_x_var_zpn,
            )
            reconst_loss += torch.mean(- x_dist_reconst.log_prob(x_target))
        reconst_loss /= self.num_sample
        
        # KL divergence between posterior and prior distributions of Z
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(z_dist_post, z_dist_prior))
        
        # Prior log probs of pitch and onset
        log_prob_pitch, log_prob_onset = self.language_model.log_prob(pitch, onset)
        log_prob_pitch = torch.mean(log_prob_pitch)
        log_prob_onset = torch.mean(log_prob_onset)

        # Additional loss for pitch and onset estimator
        pitch_loss = self.pitch_loss_fn(pitch_logits.transpose(1, 2), pitch)
        onset_loss = self.onset_loss_fn(onset_logits, onset)

        return reconst_loss + kl_loss - log_prob_pitch - log_prob_onset + pitch_loss + onset_loss

    else:
        # p(Z): prior distribution of Z
        # q(Z|X): posterior distribution of Z
        q_z_mean_x, q_z_var_x = self.z_est_model(x_input, tatum_frames)
        batch_size, num_tatums, dim_z = q_z_mean_x.shape
        z_dist_prior = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = torch.zeros_like(q_z_mean_x), 
            covariance_matrix = torch.eye(dim_z)[None, None, :].repeate([batch_size, num_tatums, 1, 1])
            )
        z_dist_post = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=q_z_mean_x, covariance_matrix=q_z_var_x
            )

        # q(P, N|X): posterior distribution of P, N
        pitch_logits, onset_logits = self.p_n_est_model(x_input, tatum_frames)
        
        # Reconstruction Error
        reconst_loss = torch.tensor(0.)
        for i in range(self.num_sample):
            z_sample = z_dist_post.rsample()
            pitch_hard = hardmax(pitch_logits, dim=-1)
            onset_hard = hardmax_bernoulli(onset_logits)
            p_x_mean_zpn, p_x_var_zpn = self.x_reconst_model(z_sample, pitch_hard, onset_hard, tatum_frames)
            x_dist_reconst = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=p_x_mean_zpn, covariance_matrix=p_x_var_zpn,
            )
            reconst_loss += torch.mean(- x_dist_reconst.log_prob(x_target))
        reconst_loss /= self.num_sample

        # Expected prior log probs w.r.t posterior distribution for P, N
        exp_log_prob_pn = self.language_model.expected_log_prob(pitch_logits, onset_logits)
        exp_log_prob_pn = torch.mean(exp_log_prob_pn)

        # KL divergence between posterior and prior distributions of Z
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(z_dist_post, z_dist_prior))
        
        # Prior log probs of pitch and onset
        log_prob_pitch, log_prob_onset = self.language_model.log_prob(pitch, onset)
        log_prob_pitch = torch.mean(log_prob_pitch)
        log_prob_onset = torch.mean(log_prob_onset)

        # Entropy of posterior distribution of P, N
        ent_pitch = ent_categorical(pitch_logits, dim=-1)
        ent_onset = ent_bernoulli(onset_logits)

        return reconst_loss - exp_log_prob_pn + kl_loss - ent_pitch - ent_onset