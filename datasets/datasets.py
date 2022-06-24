import numpy as np
import torch, torchaudio
import librosa
import pickle
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

seed = 65324

def vanila_transform(waveform, sr):
    return waveform[None, :]
    
class ConvFilter:
    def __init__(self, half_length=5, v_max=1):
        self.padding = half_length - 1
        self.conv_weights = self._get_conv_weight(half_length, v_max)

    def __call__(self, x):
        # Apply the filter on the last dimension
        # x: (batch_size, in_channel=1, length)
        # output: (batch_size, out_channel=1, length)

        return torch.nn.functional.conv1d(x, self.conv_weights, padding=self.padding)
    
    def _get_conv_weight(self, half_length, v_max):
        weights = np.linspace(0, v_max, half_length)
        weights = np.append(weights, weights[-2::-1])[None, None, :]
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, indexes_path, hdf5s_dir, 
        transform=vanila_transform,
        target_transform=vanila_transform,
        name="RWC",
        pitch_shift=False,
        separated=False,
        separated_and_mix=False,
        with_offset=False,
        onset_filter=None,
        pitch_filter=None,
        min_pitch_label=0,
        max_pitch_label=128,
        **args):

        super().__init__()
        self.name = name
        self.indexes = pickle.load(open(indexes_path, 'rb'))

        self.hdf5s_dir = hdf5s_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.configs = {
            'pitch_shift': pitch_shift,
            'separated': separated,
            'separated_and_mix': separated_and_mix,
        }

        self.onset_filter = ConvFilter(**onset_filter) if onset_filter is not None else None
        self.pitch_filter = ConvFilter(**pitch_filter) if pitch_filter is not None else None
        self.rand = np.random.RandomState(seed)

    def __len__(self):

        if self.configs['pitch_shift']:
            return len(self.indexes) * 25
        else:
            return len(self.indexes)

    def __getitem__(self, i):
        # start_time = time.time()
        index_id = i // 25 if self.configs['pitch_shift'] else i
        track_id = self.indexes[index_id]['track_id']
        frames = None
        tatum_ids = self.indexes[index_id]['tatum_ids']
        text = " "
        pitch_offset = 0
        hdf5_path = os.path.join(self.hdf5s_dir, f"p{track_id:0>3}.h5")
        hdf5s_dir_sep = self.hdf5s_dir[:self.hdf5s_dir.rfind(".")] + "_separated" + self.hdf5s_dir[self.hdf5s_dir.rfind("."):]
        hdf5_path_sep = os.path.join(hdf5s_dir_sep, f"p{track_id:0>3}.h5")

        with h5py.File(hdf5_path, 'r') as hf:
            with h5py.File(hdf5_path_sep, 'r') as hf_sep:
                waveform_mix, waveform_sep, pitches, onsets, sr, hop_length, tatum_time = self._get_data_from_hdf5(hf, pitch_offset, tatum_ids, hf_sep)
        
        # input features:
        input_features = self._get_transformed_features(waveform_mix, waveform_sep, sr, self.transform)
        target_features = self._get_transformed_features(waveform_mix, waveform_sep, sr, self.target_transform)

        # tatum frames:
        tatum_frames = librosa.time_to_frames(tatum_time, sr=sr, hop_length=hop_length)
        n_frames = tatum_frames[-1]
        # print(f"getitem using time {time.time() - start_time}")

        # targets:
        return (
            input_features[..., :n_frames], target_features[..., :n_frames], pitches, onsets,
            torch.tensor(tatum_frames, dtype=torch.long),
            )

    def _get_data_from_hdf5(self, hf, pitch_offset, tatum_ids, hf_sep):
        sr = hf.attrs['sample_rate']
        hop_length = hf.attrs['hop_length']
        pitch_shift = list(range(-12, 13))[i % 25] if self.configs['pitch_shift'] else 0

        # waveform:
        samples = librosa.time_to_samples(hf['tatum_time'][tatum_ids], sr=sr)
        frames = librosa.time_to_frames(hf['tatum_time'][tatum_ids], sr=sr, hop_length=hop_length)
        if pitch_shift == 0:
            waveform_mix = hf['waveform'][samples[0]: samples[1]]
            waveform_sep = hf_sep['waveform'][samples[0]: samples[1]]
        else:
            waveform_mix = hf[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]
            waveform_sep = hf_sep[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]

        # if len(waveform) < samples[1] - samples[0]:
        #     waveform = np.pad(waveform, (0, samples[1] - samples[0]))
        
        # tatum:
        tatum_time = hf['tatum_time'][tatum_ids[0]: tatum_ids[1] + 1]
        tatum_time -= tatum_time[0]

        # targets:

        pitches = hf['pitch_tatums'][tatum_ids[0]: tatum_ids[1]]
        pitches[pitches != 128] += pitch_shift + pitch_offset
        pitches = np.clip(pitches, 0, 128)
        pitches = self._pitch_to_zeroone(pitches)
        onsets = hf['onset_tatums'][tatum_ids[0]: tatum_ids[1]]
            
        pitches = torch.tensor(pitches, dtype=torch.float)
        onsets = torch.tensor(onsets, dtype=torch.float)

        return waveform_mix, waveform_sep, pitches, onsets, sr, hop_length, tatum_time

    def _pitch_to_zeroone(self, pitches):
        new_pitches = np.zeros((129, len(pitches)))
        new_pitches[pitches, range(len(pitches))] = 1
        return new_pitches
    
    def _get_transformed_features(self, waveform_mix, waveform_sep, sr, transform):
        if self.configs['separated_and_mix']:
            features_sep = torch.tensor(transform(waveform_sep, sr), dtype=torch.float)
            features_mix = torch.tensor(transform(waveform_mix, sr), dtype=torch.float)
            if features_sep.ndim == 3:
                features = torch.cat([features_sep, features_mix], dim=0)
            else:
                features = torch.stack([features_sep, features_mix])
        else:
            waveform = waveform_sep if self.configs['separated'] else waveform_mix
            features = torch.tensor(transform(waveform, sr), dtype=torch.float)

        return features