import torch, numpy as np
from .utils import *

class CrossEntropyLossWithProb:
    
    def __init__(self, weight=None, ignore_index=None):
        self.weight = weight
        self.ignore_index = ignore_index
    
    def to(self, *args, **kwargs):
        if self.weight is not None:
            self.weight = self.weight.to(*args, **kwargs)
        return self
    
    def __call__(self, inputs, targets):
        # inputs: (N, ..., C)
        # targets: (N, ..., C)
        inputs = torch.log_softmax(inputs, dim=-1)
        loss = inputs * targets
        if self.weight is not None:
            loss = (self.weight * loss)
        loss = - torch.sum(loss, dim=-1)
        return torch.mean(loss)

class VAE:
    def __init__(self, z_est_model, p_n_est_model, x_reconst_model, language_model, num_sample=1, loss_weights=None):
        """ 
        z_est_model:
          Input: 
            x_input: (batch_size, num_channels, num_features, num_frames)
            tatum_frames: (batch_size, num_tatums + 1)
          Output:
            q_z_mean_x: (batch_size, num_frames, dim_z)
            q_z_std_x: (batch_size, num_frames, dim_z)
        
        p_n_est_model:
          Input:
            x_input: (batch_size, num_channels, num_features, num_frames)
            tatum_frames: (batch_size, num_tatums + 1)
          Output:
            pitches_logits: (batch_size, num_tatums, num_pitches)
            onsets_logits: (batch_size, num_tatums)

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
        self.p_n_est_model = p_n_est_model
        self.x_reconst_model = x_reconst_model
        self.language_model = language_model
        self.num_sample = num_sample
        self.pitch_loss_fn = CrossEntropyLossWithProb()
        self.onset_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_weights = loss_weights

    def __call__(
        self,
        x_input,
        x_target,
        input_lengths,
        tatum_frames,
        supervised=False,
        pitch=None, 
        onset=None,
        ):
        """
        Input:
          x_input: (batch_size, num_channels, num_features, num_frames)
          x_target: (batch_size, num_frames, num_features)
          input_lengths: (batch_size)
          tatum_frames: (batch_size, num_tatumes)
          pitch: (batch_size, num_tatums, num_pitches)
          onset: (batch_size, num_tatums)

        Output:
          loss: torch.Tensor
        """
        if supervised:
            if pitch is None or onset is None:
                raise ValueError("For supervised learning, pitch and onset labels must be given!")
        
            # p(Z): prior distribution of Z
            # q(Z|X): posterior distribution of Z
            q_z_mean_x, q_z_std_x = self.z_est_model(x_input, tatum_frames)
            batch_size, num_tatums, dim_z = q_z_mean_x.shape

            z_dist_post = torch.distributions.normal.Normal(
                loc=q_z_mean_x, scale=q_z_std_x
                )
            
            reconst_loss = []
            # Reconstruction Error
            for i in range(self.num_sample):
                z_sample = z_dist_post.rsample()
                p_x_mean_zpn, p_x_std_zpn = self.x_reconst_model(z_sample, pitch, onset, tatum_frames)
                for j in range(batch_size):
                    x_dist_reconst = torch.distributions.normal.Normal(
                    loc=p_x_mean_zpn[j, :input_lengths[j], :], scale=p_x_std_zpn[j, :input_lengths[j], :],
                    )
                    reconst_loss.append(torch.mean(- x_dist_reconst.log_prob(x_target[j, :input_lengths[j], :])))
                    # print(x_dist_reconst, x_target[j, :input_lengths[j], :].shape)
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

            # Prior log probs of pitch and onset
            log_prob_pitch, log_prob_onset = self.language_model.log_prob(pitch, onset)
            log_prob_pitch = torch.mean(log_prob_pitch)
            log_prob_onset = torch.mean(log_prob_onset)
            
            # q(P, N|X): posterior distribution of P, N
            pitch_logits, onset_logits = self.p_n_est_model(x_input, tatum_frames)
            # Additional loss for pitch and onset estimator
            pitch_loss = self.pitch_loss_fn(pitch_logits, pitch)
            onset_loss = self.onset_loss_fn(onset_logits, onset)    
            return reconst_loss + kl_loss - log_prob_pitch - log_prob_onset + pitch_loss + onset_loss
    
        else:
            # p(Z): prior distribution of Z
            # q(Z|X): posterior distribution of Z
            q_z_mean_x, q_z_std_x = self.z_est_model(x_input, tatum_frames)
            batch_size, num_tatums, dim_z = q_z_mean_x.shape
            z_dist_post = torch.distributions.normal.Normal(
                loc=q_z_mean_x, scale=q_z_std_x
                )
            
            # q(P, N|X): posterior distribution of P, N
            pitch_logits, onset_logits = self.p_n_est_model(x_input, tatum_frames)
            
            # Reconstruction Error
            reconst_loss = []
            for i in range(self.num_sample):
                z_sample = z_dist_post.rsample()
                pitch_hard = hardmax(pitch_logits, dim=-1)
                onset_hard = hardmax_bernoulli(onset_logits)
                p_x_mean_zpn, p_x_std_zpn = self.x_reconst_model(z_sample, pitch_hard, onset_hard, tatum_frames)
                for j in range(batch_size):
                    x_dist_reconst = torch.distributions.normal.Normal(
                    loc=p_x_mean_zpn[j, :input_lengths[j], :], scale=p_x_std_zpn[j, :input_lengths[j], :],
                    )
                    reconst_loss.append(torch.mean(- x_dist_reconst.log_prob(x_target[j, :input_lengths[j], :])))

            reconst_loss = torch.mean(torch.stack(reconst_loss))
            
            # Expected prior log probs w.r.t posterior distribution for P, N
            exp_log_prob_pn = self.language_model.expected_log_prob(pitch_logits, onset_logits)
            exp_log_prob_pn = torch.mean(exp_log_prob_pn)
            
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
            
            # Entropy of posterior distribution of P, N
            ent_pitch = torch.mean(ent_categorical(pitch_logits, dim=-1))
            ent_onset = torch.mean(ent_bernoulli(onset_logits))
            
            return reconst_loss - exp_log_prob_pn + kl_loss - ent_pitch - ent_onset