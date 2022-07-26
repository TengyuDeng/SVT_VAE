
import torch, numpy as np

def get_collate_fn(collate_fn_type="tatum_based"):
    if collate_fn_type == "tatum_based":
        return collate_fn_tatum_based
    else:
        return collate_fn_frame_based
        
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

    features_list, pitches, onsets, tatums, texts = tuple(zip(*list_data))

    # features
    
    features_list = list(zip(*features_list))
    for i in range(len(features_list)):
        features_list[i] = [feature.transpose(0, -1) for feature in features_list[i]]

    input_lengths = torch.tensor([feature.shape[0] for feature in features_list[0]], dtype=torch.int)
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.int)

    input_features = [
        torch.nn.utils.rnn.pad_sequence(features, batch_first=True).transpose(1, -1)
        for features in features_list
        ]
    
    pitches = torch.stack(pitches)
    onsets = torch.stack(onsets)
    texts = torch.cat(texts)
    
    tatum_frames = torch.stack(tatums)

    return (input_features, pitches, onsets, input_lengths, tatum_frames, texts, text_lengths)

def collate_fn_frame_based(list_data):
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

    features_list, pitches, onsets, tatums, texts = tuple(zip(*list_data))

    # features
    
    features_list = list(zip(*features_list))
    for i in range(len(features_list)):
        features_list[i] = [feature.transpose(0, -1) for feature in features_list[i]]
    if pitches[0].ndim == 2:
        pitches = [pitch.transpose(0, -1) for pitch in pitches]

    input_lengths = torch.tensor([feature.shape[0] for feature in features_list[0]], dtype=torch.int)
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.int)

    input_features = [
        torch.nn.utils.rnn.pad_sequence(features, batch_first=True).transpose(1, -1)
        for features in features_list
        ]
    
    if new_pitches[0].ndim == 1:
        pitches = torch.nn.utils.rnn.pad_sequence(new_pitches, batch_first=True, padding_value=128)
    else:
        pitches = torch.nn.utils.rnn.pad_sequence(new_pitches, batch_first=True, padding_value=0).transpose(1, -1)
    onsets = torch.nn.utils.rnn.pad_sequence(new_onsets, batch_first=True, padding_value=0.)

    texts = torch.cat(texts)
    
    tatum_frames = torch.stack(tatums)

    return (input_features, pitches, onsets, input_lengths, tatum_frames, texts, text_lengths)