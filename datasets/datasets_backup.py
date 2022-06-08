import os
import sys
sys.path.append("..")

import numpy as np
import torch, torchaudio
import librosa
import pickle
import h5py
from decoders import Mycodec
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

seed = 65324

def get_conv_weight(half_length=5, v_max=1):
    weights = np.linspace(0, v_max, half_length)
    weights = np.append(weights, weights[-2::-1])[None, None, :]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

class myDataset(torch.utils.data.Dataset):
    
    def __init__(self, name, indexes_path, hdf5s_dir, transform,
        text_only=False, 
        frame_based=False, 
        pitch_shift=False, 
        separated=False,
        with_offset=False,
        tri_window_size=0,
        **args):

        super(myDataset, self).__init__()
        self.name = name
        self.indexes = pickle.load(open(indexes_path, 'rb'))
        self.hdf5s_dir = hdf5s_dir
        self.target_type = "word"
        self.codec = Mycodec(self.target_type)
        self.transform = transform

        self.text_only = text_only
        self.frame_based = frame_based
        self.pitch_shift = pitch_shift
        self.separated = separated
        self.with_offset = with_offset
        self.tri_window_size = tri_window_size
        self.onset_conv_weights = get_conv_weight(half_length=tri_window_size) if tri_window_size > 0 else None
        self.rand = np.random.RandomState(seed)

    def __len__(self):

        if self.name == "RWC" and self.pitch_shift:
            return len(self.indexes) * 25
        else:
            return len(self.indexes)

    def __getitem__(self, i):
        # start_time = time.time()
        if self.name == "RWC":
            index_id = i // 25 if self.pitch_shift else i
            track_id = self.indexes[index_id]['track_id']
            tatum_ids = self.indexes[index_id]['tatum_ids']
            hdf5_path = os.path.join(self.hdf5s_dir, f"p{track_id:0>3}.h5")
        
        elif self.name == "DALI":
            dali_id = self.indexes[i]['dali_id']
            tatum_ids = self.indexes[i]['tatum_ids']
            text = self.indexes[i]['text']
            pitch_offset = self.indexes[i]['offset']
            hdf5_path = os.path.join(self.hdf5s_dir, f"{dali_id}.h5")

        else:
            raise RuntimeError("Task not supported!")
        
        if self.target_type == "phoneme":
            text = self.codec.phonemize(text)
        text = self.codec.encode(text)

        with h5py.File(hdf5_path, 'r') as hf:
            sr = hf.attrs['sample_rate']
            hop_length = hf.attrs['hop_length']
            pitch_shift = list(range(-12, 13))[i % 25] if self.pitch_shift else 0

            # waveform:
            samples = librosa.time_to_samples(hf['tatum_time'][tatum_ids], sr=sr)
            frames = librosa.time_to_frames(hf['tatum_time'][tatum_ids], sr=sr, hop_length=hop_length)
            if self.name == "RWC":
                if pitch_shift == 0:
                    waveform = hf['waveform'][samples[0]: samples[1]]
                else:
                    waveform = hf[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]
            
            elif self.name == "DALI":
                if self.separated:
                    waveform = hf['waveform_separated'][samples[0]: samples[1]]
                else:
                    waveform = hf['waveform'][samples[0]: samples[1]]

            if len(waveform) < samples[1] - samples[0]:
                waveform = np.pad(waveform, (0, samples[1] - samples[0]))
            
            # tatum:
            tatum_time = hf['tatum_time'][tatum_ids[0]: tatum_ids[1] + 1]
            tatum_time -= tatum_time[0]
            
            # targets:
            if self.frame_based:
                midi_notes = hf['midi_notes'][:] if self.name == "RWC" else hf['annot_notes'][:]
            else:
                pitches = hf['pitch_tatums'][tatum_ids[0]: tatum_ids[1]]
                pitches[pitches != 128] += pitch_shift
                if self.name == "DALI":
                    pitches[pitches != 128] += pitch_offset
                pitches = np.clip(pitches, 0, 128)
                onsets = hf['onset_tatums'][tatum_ids[0]: tatum_ids[1]]
        
        # Input features:
        input_features = self.transform(waveform, sr)
        
        # tatum frames:
        tatum_frames = librosa.time_to_frames(tatum_time, sr=sr, hop_length=hop_length)
        if tatum_frames[-1] > input_features.shape[-1]:
            print(f"samples:{samples}, waveform:{waveform.shape} input_features:{input_features.shape}, the last tatum: [{tatum_frames[-2]}: {tatum_frames[-1]}]")
        # print(f"getitem using time {time.time() - start_time}")
        
        # targets:
        if self.text_only:
            return (
            torch.tensor(input_features, dtype=torch.float),
            torch.tensor(text, dtype=torch.int),
            )
        elif self.frame_based:
            frame_length = input_features.shape[-1]
            pitches_frame = np.full(frame_length, 128, dtype=int)
            onsets_frame = np.full(frame_length, 0, dtype=float)
            for note in midi_notes:
                start_frame, end_frame = librosa.time_to_frames(note[:2], sr=sr, hop_length=hop_length)
                if start_frame >= frames[0] and start_frame < frames[1]:
                    start_frame -= frames[0]
                    end_frame -= frames[0]
                    pitches_frame[start_frame:end_frame] = note[-1]
                    onsets_frame[start_frame] = 1
                    if self.with_offset and end_frame < frame_length:
                        onsets_frame[end_frame] = 1
            pitches_frame[pitches_frame != 128] += pitch_shift
            if self.name == "DALI":
                    pitches_frame[pitches_frame != 128] += pitch_offset
            pitches_frame = np.clip(pitches_frame, 0, 128)
            onsets = torch.tensor(onsets_frame, dtype=torch.float)
            if self.onset_conv_weights is not None:
                onsets = torch.nn.functional.conv1d(onsets[None, None,:], self.onset_conv_weights, padding=self.tri_window_size-1).squeeze()
            return (
                torch.tensor(input_features, dtype=torch.float),
                torch.tensor(pitches_frame, dtype=torch.long),
                onsets,
                torch.tensor(text, dtype=torch.int),
                torch.tensor(tatum_frames, dtype=torch.long),
                )
        
        else:
            return (
                torch.tensor(input_features, dtype=torch.float), 
                torch.tensor(pitches, dtype=torch.long), 
                torch.tensor(onsets, dtype=torch.float),
                torch.tensor(text, dtype=torch.int),
                torch.tensor(tatum_frames, dtype=torch.long),
                )


class myLibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, urls=["train-clean-360"], text_only=True, **args):
        super(myLibriSpeechDataset, self).__init__()
        self.LibriSpeechDataset = torch.utils.data.ConcatDataset([
            torchaudio.datasets.LIBRISPEECH(
                root=root,
                url=url,
                download=True,
            ) for url in urls
        ])
        self.target_type = "word"
        self.codec = Mycodec(self.target_type)
        self.text_only = text_only
        self.transform = transform

    def __len__(self):
        return len(self.LibriSpeechDataset)

    def __getitem__(self, i):
        # waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[self.indexes[i]['index']]
        waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[i]
        text = text.lower()
        
        waveform = waveform.squeeze().numpy()
        input_features = self.transform(waveform, sr)
        
        if self.target_type == "phoneme":
            text = self.codec.phonemize(text)
        text = self.codec.encode(text)
        
        if self.text_only:
            return (
                torch.tensor(input_features, dtype=torch.float),
                torch.tensor(text, dtype=torch.int),
                )
        else:
            frame_length = input_features.shape[-1]
            pitches_frame = np.full((frame_length,), 128)
            onsets_frame = np.zeros((frame_length,))
            tatum_frames = [0,frame_length]
            return (
                torch.tensor(input_features, dtype=torch.float),
                torch.tensor(pitches_frame, dtype=torch.long),
                torch.tensor(onsets_frame, dtype=torch.float),
                torch.tensor(text, dtype=torch.int),
                torch.tensor(tatum_frames, dtype=torch.long),
                )

# class MyBatchSampler:

#     def __init__(self, cumulative_sizes, batch_size, percentage):
#         self.cumulative_sizes = cumulative_sizes
#         self.batch_size_a = int(batch_size * percentage)
#         self.batch_size_b = batch_size - self.batch_size_a

#     def __len__(self):
#         return self.cumulative_sizes[0] // self.batch_size_a

#     def __iter__(self):
#         indexes_a = list(np.random.permutation(range(0, self.cumulative_sizes[0])))
#         indexes_b = list(np.random.permutation(range(self.cumulative_sizes[0], self.cumulative_sizes[1])))
        
#         i = 0
#         j = 0
#         while i < len(indexes_a):
#             if i + self.batch_size_a >= len(indexes_a):
#                 break
#             if j + self.batch_size_b >= len(indexes_b):
#                 j = 0
            
#             indexes = indexes_a[i : i + self.batch_size_a] + indexes_b[j : j + self.batch_size_b]
#             yield indexes

#             i += self.batch_size_a
#             j += self.batch_size_b


# def get_mix_dataloaders(configs):
#     num_workers = 8
    
#     batch_size = configs['batch_size']
#     percentage = configs['percentage_of_DALI']

#     hdf5s_dir = configs['hdf5s_dir']
#     indexes_dir = configs['indexes_dir']

#     train_idx_path = os.path.join(indexes_dir, "train_idx.pkl")
#     test_idx_path = os.path.join(indexes_dir, "test_idx.pkl")
#     separated = configs['separated'] if 'separated' in configs else False
    
#     configs_DALI = configs.copy()
#     configs_LibriSpeech = configs.copy()
#     configs_DALI['mel_paras'] = configs_DALI['mel_paras_DALI']
#     configs_LibriSpeech['mel_paras'] = configs_LibriSpeech['mel_paras_LibriSpeech']
#     if 'urls' in configs:
#         libri_dataset = myLibriSpeechDataset(
#             root="/n/work1/deng/data/",
#             urls=configs['urls'],
#             configs=configs,
#         )
#     else:
#         libri_dataset = myLibriSpeechDataset(
#             root="/n/work1/deng/data/",
#             url=configs['url'],
#             configs=configs,
#         )
#     train_dataset = torch.utils.data.ConcatDataset([
#         myDataset(
#         indexes_path=train_idx_path,
#         hdf5s_dir=hdf5s_dir,
#         task="DALI",
#         configs=configs_DALI,
#         separated=separated,
#         ),
#         libri_dataset,
#         ])
#     test_datasets = [
#         myDataset(
#         indexes_path=test_idx_path,
#         hdf5s_dir=hdf5s_dir,
#         task="DALI",
#         configs=configs_DALI,
#         separated=separated,
#         ),
#         myLibriSpeechDataset(
#         root="/n/work1/deng/data/",
#         url="dev-other",
#         configs=configs_LibriSpeech,
#         ),
#         ]

#     train_dataloader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         num_workers=num_workers,
#         collate_fn=collate_fn_text_only,
#         batch_sampler=MyBatchSampler(train_dataset.cumulative_sizes, batch_size, percentage),
#         pin_memory=True,
#         )
#     test_dataloaders = [
#     torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         num_workers=num_workers,
#         collate_fn=collate_fn_text_only,
#         batch_size=batch_size,
#         shuffle=True,
#         pin_memory=True,
#         )
#     for test_dataset in test_datasets
#     ]

#     return train_dataloader, test_dataloaders
