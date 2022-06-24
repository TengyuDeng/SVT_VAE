
import torch, numpy as np

def get_collate_fn(collate_fn_type="tatum_based"):
    if collate_fn_type == "tatum_based":
        return collate_fn_tatum_based
    else:
        return collate_fn_tatum_based
        
def collate_fn_tatum_based(list_data):
    # Input:
    #     list_data: [
    #         (feature1, target1),
    #         (feature2, target2),
    #     ]
    # Output:
    #     (audio_features, targets, input_lengths, target_lengths)
    #     audio_features: (batch_size, feature_num, time)
    #     targets: (\sum{lengths})
    
    # start_time = time.time()

    raw_features, raw_targets, raw_pitches, raw_onsets, raw_tatums = tuple(zip(*list_data))

    # features
    new_features = []
    new_targets = []
    new_pitches = []
    new_onsets = []
    input_lengths = []
    new_tatums = []

    for n in range(len(list_data)):
        feature = raw_features[n]
        target_feature = raw_targets[n]
        pitch = raw_pitches[n]
        onset = raw_onsets[n]
        tatum = raw_tatums[n]
        # feature (channel, time)
        if feature.ndim > 1:
            new_features.append(feature.transpose(0, -1))
            new_targets.append(target_feature.transpose(0, -1))
            input_lengths.append(feature.shape[-1])
            new_pitches.append(pitch)
            new_onsets.append(onset)
            new_tatums.append(tatum)
    
    input_features = torch.nn.utils.rnn.pad_sequence(new_features, batch_first=True).transpose(1, -1)
    target_features = torch.nn.utils.rnn.pad_sequence(new_targets, batch_first=True).transpose(1, -1)
    pitches = torch.stack(new_pitches)
    onsets = torch.stack(new_onsets)
    input_lengths = torch.tensor(input_lengths, dtype=torch.int)
    tatum_frames = torch.stack(new_tatums)

    return (input_features, target_features, pitches, onsets, input_lengths, tatum_frames)
