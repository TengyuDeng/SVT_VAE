import os
import yaml
import numpy as np
import torch
from torch import nn
import librosa
# from g2p_en import G2p
import textdistance
import mir_eval

def read_yaml(config_yaml: str):

    with open(config_yaml, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    return configs

def downsample_length(orig_length, down_sample):
    if isinstance(orig_length, torch.Tensor):
        return torch.div(orig_length - 1, down_sample, rounding_mode='floor') + 1
    else:
        return (orig_length - 1) // down_sample + 1

def remove_space(lst):
    return list(filter(lambda x:x!=' ', lst))
    
def mycer(ref, pre):
    ref = ref.tolist()
    pre = pre.tolist()
    if len(ref) == 0:
        return 0
    else:
        return textdistance.levenshtein.distance(ref, pre) / len(ref)

def freq2pitch(freq, datatype="np"):
    if datatype == "np":
        return int(np.round(12 * (np.log2(freq) - np.log2(440.)))) + 69
    else:
        return int(torch.round(12 * (torch.log2(freq) - torch.log2(440.)))) + 69

def pitch2freq(class_id):

    return (440. * 2 ** ((class_id - 69) / 12)) * (class_id != 128) + (1e-3) * (class_id == 128)

# Sampling with discrete distributions (hard)
def hardmax(logits, dim=-1):
    index = logits.argmax(dim=dim, keepdim=True)
    y_soft = torch.nn.functional.softmax(logits, dim=dim)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft

def hardmax_bernoulli(logits, threshold=0.5):
    y_soft = torch.sigmoid(logits)
    y_hard = torch.zeros_like(logits)
    y_hard[y_soft > threshold] = 1.
    return y_hard - y_soft.detach() + y_soft

def adjust_shape(x, ref, ref_shape=None):
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
    if ref_shape is not None:
        H_ref, W_ref = ref_shape
    else:
        H_ref, W_ref = ref.shape[-2:]
    if (H_x - H_ref) * (W_x - W_ref) < 0:
        raise ValueError(f"Wrong shapes: {x.shape}, {ref.shape}")
    
    if H_x >= H_ref:
        return x[..., :H_ref, :W_ref]
    else:
        padding = (0, W_ref - W_x, 0, H_ref - H_x)
        return nn.ZeroPad2d(padding)(x)

# Upsampling
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

# Evaluate based on frames
def tatums_to_frames(feature_tatums, tatum_frames, frame_length=None, fill_value=0):
    feature_tatums = np.array(feature_tatums)
    assert len(feature_tatums) < len(tatum_frames)
    if frame_length is None:
        frame_length = tatum_frames[-1]
    feature_frames = np.full(frame_length, fill_value, dtype=feature_tatums.dtype)
    for i in range(len(feature_tatums)):
        feature_frames[tatum_frames[i]:tatum_frames[i+1]] = feature_tatums[i]
    return feature_frames

def evaluate_frames(pitches_ref, pitches_pre):
    if len(pitches_ref) != len(pitches_pre):
        length = min(len(pitches_ref), len(pitches_pre))
        pitches_ref = pitches_ref[:length]
        pitches_pre = pitches_pre[:length]
    err = np.sum(pitches_ref != pitches_pre) / len(pitches_ref)
    return err

def evaluate_frames_batch(pitches_ref, pitches_pre, input_lengths):
    err = 0
    for i in range(len(input_lengths)):
        err += evaluate_frames(pitches_ref[i, :input_lengths[i]], pitches_pre[i, :input_lengths[i]])
    err /= len(input_lengths)
    return err

def evaluate_melody(pitches_ref, pitches_pre, cent_tolerance=50):
    # pitches_ref: (129, frames)
    # pitches_pre: (129, frames)
    
    # formalize the reference pitch series
    values_ref = np.argmax(pitches_ref[:128, :], axis=0).astype(float)
    values_ref *= 100
    voicing_ref = ((1 - pitches_ref[-1, :]) >= 0.5).astype(int)
    
    # formalize the prediction pitch series
    values_pre = np.argmax(pitches_pre[:128, :], axis=0).astype(float)
    values_pre *= 100
    voicing_pre = ((1 - pitches_pre[-1, :]) >= 0.5).astype(int)
    
    # evaluate
    vr, vfa = mir_eval.melody.voicing_measures(voicing_ref, voicing_pre)
    rpa = mir_eval.melody.raw_pitch_accuracy(voicing_ref, values_ref, voicing_pre, values_pre, cent_tolerance=cent_tolerance)
    rca = mir_eval.melody.raw_chroma_accuracy(voicing_ref, values_ref, voicing_pre, values_pre, cent_tolerance=cent_tolerance)
    oa = mir_eval.melody.overall_accuracy(voicing_ref, values_ref, voicing_pre, values_pre, cent_tolerance=cent_tolerance)
    
    return [vr, vfa, rpa, rca, oa]

def evaluate_melody_batch(pitches_ref, pitches_pre, input_lengths, cent_tolerance=50):
    statistics = []
    for i in range(len(input_lengths)):
        statistics.append(evaluate_melody(pitches_ref[i, :input_lengths[i]], pitches_pre[i, :input_lengths[i]], cent_tolerance=cent_tolerance))
    return np.mean(statistics, axis=0)

# Evaluate based on notes
# def activs_to_onsets_offsets(activs):
#     activs = np.append(0, activs)
#     activs_diff = np.diff(np.append(activs, 0))
#     onsets = np.argwhere(activs_diff == 1).flatten()
#     offsets = np.argwhere(activs_diff == -1).flatten()
#     assert onsets.shape == offsets.shape
#     return onsets, offsets

def peakpicking(activs, window_size=2, threshold=0.4):
    peaks = np.zeros(activs.shape, dtype=np.int)
    
    for i in range(activs.shape[-1]):
        if i < window_size:
            window = activs[..., :i + window_size + 1]
        elif i > activs.shape[-1] - window_size - 1:
            window = activs[..., i - window_size:]
        else:
            window = activs[..., i - window_size: i + window_size + 1]
        peaks[..., i] = ((activs[..., i] == np.max(window, axis=-1)) * (activs[..., i] > threshold)).astype(int)
        
    return peaks

def formalize_to_intervals(pitches, onsets, offsets=None, tatum_frames=None, sr=22050, hop_length=256, with_empty=False):
    # onsets: (T,)
    # pitches: (T, )
    # offsets: (T,)
    # tatum_frames: (T + 1,)
    pitch_change = np.diff(np.append(0, pitches))
    onset_tatums = np.argwhere((pitch_change != 0) + (onsets == 1)).flatten()
    if len(onset_tatums) == 0:
        return np.empty(shape=(0,2)), np.empty(shape=0)
    if offsets is not None:
        offset_tatums = np.argwhere(offsets == 1).flatten()
    else:
        offset_tatums = np.append(onset_tatums[1:], len(onsets))
    
    # note_interval_tatums: (T, 2)
    note_interval_tatums = np.stack([onset_tatums, offset_tatums], axis=1)
    if tatum_frames is not None:
        note_interval_frames = tatum_frames[note_interval_tatums]
    else:
        note_interval_frames = note_interval_tatums
    
    assert pitches.dtype == int, pitches.dtype
    pitch_in_notes = []
    for i in range(len(note_interval_tatums)):
        start_tatum = note_interval_tatums[i, 0]
        end_tatum = note_interval_tatums[i, 1]
        pitches_in_interval = pitches[start_tatum: end_tatum]
        pitch = np.argmax(np.bincount(pitches_in_interval))
        pitch_in_notes.append(pitch)

    note_interval_times = librosa.frames_to_time(note_interval_frames, sr=sr, hop_length=hop_length)
    pitch_in_notes = np.array(pitch_in_notes)
    if with_empty:
        return note_interval_times, pitch_in_notes
    else:
        return note_interval_times[pitch_in_notes != 128, :], pitch_in_notes[pitch_in_notes != 128]

def evaluate_notes(pitches, onsets, pitches_pre, onsets_pre, input_lengths=None, tatum_frames=None, **kwargs):
    if input_lengths is None:
        input_lengths = np.full((pitches.shape[0],), fill_value=pitches.shape[1])
    if tatum_frames is None:
        tatum_frames = [None] * len(input_lengths)
    rets = []
    for i in range(len(input_lengths)):
        note_interval_times, note_values = formalize_to_intervals(
            pitches[i, :input_lengths[i]], 
            onsets[i, :input_lengths[i]],
            tatum_frames=tatum_frames[i],
            **kwargs
            )
        note_interval_times_pre, note_values_pre = formalize_to_intervals(
            pitches_pre[i, :input_lengths[i]], 
            onsets_pre[i, :input_lengths[i]], 
            tatum_frames=tatum_frames[i],
            **kwargs
            )
        note_freqs = pitch2freq(note_values)
        note_freqs_pre = pitch2freq(note_values_pre)
        score = mir_eval.transcription.evaluate(note_interval_times, note_freqs, note_interval_times_pre, note_freqs_pre)
    
        ret = np.zeros(9)
        ret[0] = score['Precision']
        ret[1] = score['Recall']
        ret[2] = score['F-measure']
        ret[3] = score['Precision_no_offset']
        ret[4] = score['Recall_no_offset']
        ret[5] = score['F-measure_no_offset']
        ret[6] = score['Onset_Precision']
        ret[7] = score['Onset_Recall']
        ret[8] = score['Onset_F-measure']
        rets.append(ret)

    rets = np.array(rets)
    rets = np.mean(rets, axis=0)

    return rets

# Model utilities
def get_padding(kernel_size):
    if isinstance(kernel_size, int):
        padding = kernel_size // 2
    else:
        padding = tuple(x // 2 for x in kernel_size)
    return padding

def get_conv_weight(half_length=5):
    weights = np.linspace(0, 1, half_length)
    weights = np.append(weights, weights[-2::-1])[None, None, :]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

def pitch_label_to_zeroone(pitches, n_classes = 129):
    # xs: [[0, 0, ..., 0], [1,1, ..., 1], ..., [batch_size, ..., batch_size]]
    # zs: [[0, 1, ..., length], [0, 1, ..., length], ..., [0, 1, ..., length]]
    xs = (torch.tensor([range(pitches.shape[0])] * pitches.shape[1]).T)[pitches < n_classes]
    zs = torch.tensor([range(pitches.shape[1])] * pitches.shape[0])[pitches < n_classes]
    
    new_pitches = torch.zeros(pitches.shape[0], n_classes, pitches.shape[1], device=pitches.device)
    new_pitches[xs, pitches[pitches < n_classes], zs] = 1
    
    return new_pitches

def upsample(x, times=2):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.stack([x] * times, axis=-1).reshape(*x.shape[:-1], x.shape[-1] * times)

def upsample_mask(k, times=2):
    x = np.identity(k, dtype=np.float32)
    return upsample(x)

def get_filter_midi_to_mel(sr=16000, n_fft=2048, n_classes=129, **args):
    fft_bins = (librosa.midi_to_hz(range(128)) / sr * n_fft).astype(int)
    trans = np.zeros((n_fft // 2 + 1, n_classes))
    trans[fft_bins[fft_bins < n_fft // 2 + 1], np.arange(128)[fft_bins < n_fft // 2 + 1]] = 1
    return librosa.filters.mel(sr=sr, n_fft=n_fft, **args) @ trans
