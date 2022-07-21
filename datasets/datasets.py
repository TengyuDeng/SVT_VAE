import numpy as np
import torch, torchaudio
import librosa
import pickle
import os, h5py
from decoders import Mycodec

# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

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
    
    def __init__(self, indexes_path, hdf5s_dir, hop_length,
        transform=vanila_transform,
        target_transform=vanila_transform,
        tatum_based=True,
        datatypes=["separated"],
        codec_type="word",
        min_pitch_label=0,
        max_pitch_label=128,
        ):

        super().__init__()
        self.indexes = pickle.load(open(indexes_path, 'rb'))
        self.hdf5s_dir = hdf5s_dir
        self.hop_length = hop_length

        self.transform = transform
        self.target_transform = target_transform
        self.tatum_based = tatum_based
        self.datatypes = datatypes
        self.codec_type = codec_type
        if self.codec_type == "with_pitch":
            self.codec = Mycodec(self.codec_type, min_pitch_label, max_pitch_label)
        else:
            self.codec = Mycodec(self.codec_type)

        self.configs = {}

        self.rand = np.random.RandomState(seed)

    def __len__(self):

        return len(self.indexes)

    def _get_index(self, i):
        pass
        # Return positions, hdf5_paths, pitch_offset, text
    
    def _get_waveform(self, hf, positions, datatype, index=0):
        pass
        # Return waveform (of datatype)

    def _get_data(self, hf, positions, pitch_offset, index=0):
        pass
        # Return pitches, onsets, sr, tatum_time
    
    def _get_transformed_features(self, waveforms, sr, transform):
        features = []
        for datatype in waveforms:
            features.append(torch.tensor(transform(waveforms[datatype], sr), dtype=torch.float))

        if len(features) > 1:
            if features[0].ndim == 3:
                return torch.cat(features, dim=0)
            else:
                return torch.stack(features)
        else:
            return features[0]

    def _create_frame_based_targets(self, midi_notes, sr, frames, pitch_shift=0, pitch_offset=0):
        frame_length = frames[1] - frames[0]
        pitches_frame = np.full(frame_length, 128, dtype=int)
        onsets_frame = np.full(frame_length, 0, dtype=float)
        for note in midi_notes:
            start_frame, end_frame = librosa.time_to_frames(note[:2], sr=sr, hop_length=self.hop_length)
            if start_frame >= frames[0] and start_frame < frames[1]:
                start_frame -= frames[0]
                end_frame -= frames[0]
                pitches_frame[start_frame:end_frame] = note[-1]
                onsets_frame[start_frame] = 1

        pitches_frame[pitches_frame != 128] += pitch_shift + pitch_offset
        pitches_frame = np.clip(pitches_frame, 0, 128)
        pitches_frame = self._pitch_to_zeroone(pitches_frame)
        
        pitches_frame = torch.tensor(pitches_frame, dtype=torch.float)
        onsets_frame = torch.tensor(onsets_frame, dtype=torch.float)

        return pitches_frame, onsets_frame

    def _pitch_to_zeroone(self, pitches):
        new_pitches = np.zeros((129, len(pitches)))
        new_pitches[pitches, range(len(pitches))] = 1
        return new_pitches
        
    def __getitem__(self, i):
        # start_time = time.time()
        
        positions, hdf5_paths, pitch_offset, text = self._get_index(i)
        if self.codec_type == "phoneme":
            text = self.codec.phonemize(text)
        text = self.codec.encode(text)

        hfs = {}
        for key in hdf5_paths:
            hfs[key] = h5py.File(hdf5_paths[key], 'r')
        waveforms = {}
        for datatype in self.datatypes:
            waveforms[datatype] = self._get_waveform(hfs[datatype], positions, datatype, index=i)
        pitches, onsets, sr, tatum_time = self._get_data(hfs['mix'], positions, pitch_offset, index=i)
        for key in hfs:
            hfs[key].close()

        # input features:
        input_features = self._get_transformed_features(waveforms, sr, self.transform)
        target_features = self._get_transformed_features(waveforms, sr, self.target_transform)

        # tatum frames:
        if self.tatum_based:
            tatum_frames = librosa.time_to_frames(tatum_time, sr=sr, hop_length=self.hop_length)
            n_frames = tatum_frames[-1]
        # print(f"getitem using time {time.time() - start_time}")
            return (
                input_features[..., :n_frames], target_features[..., :n_frames], pitches, onsets,
                torch.tensor(tatum_frames, dtype=torch.long),
                torch.tensor(text, dtype=torch.int),
                )
        else:
            n_frames = min(input_features.shape[-1], pitches.shape[0])
            tatum_frames = torch.tensor([0, n_frames], dtype=torch.long)

            return (
                input_features[..., :n_frames], target_features[..., :n_frames], pitches[..., :n_frames], onsets[:n_frames],
                torch.tensor(tatum_frames, dtype=torch.long),
                torch.tensor(text, dtype=torch.int),
                )

class RWCDataset(MyDataset):
    
    def __init__(self, indexes_path, hdf5s_dir, 
        pitch_shift=False,
        **args,
        ):
        # *** self.configs['pitch_shift']
        # 
        super().__init__(indexes_path, hdf5s_dir, **args)
        
        self.configs = {
            'pitch_shift': pitch_shift,
        }

    def __len__(self):

        if self.configs['pitch_shift']:
            return len(self.indexes) * 25
        else:
            return len(self.indexes)
    
    def _get_index(self, i):
        index_id = i // 25 if self.configs['pitch_shift'] else i
        track_id = self.indexes[index_id]['track_id']
        if self.tatum_based:
            positions = self.indexes[index_id]['tatum_ids']
        else:
            positions = self.indexes[index_id]['frames']
        pitch_offset = 0
        text = " "

        hdf5s_dir_sep = self.hdf5s_dir[:self.hdf5s_dir.rfind(".")] + "_separated" + self.hdf5s_dir[self.hdf5s_dir.rfind("."):]
        hdf5s_dir_vd = self.hdf5s_dir[:self.hdf5s_dir.rfind(".")] + "_vd" + self.hdf5s_dir[self.hdf5s_dir.rfind("."):]
        hdf5_paths = {
            'mix': os.path.join(self.hdf5s_dir, f"p{track_id:0>3}.h5"),
            'separated': os.path.join(hdf5s_dir_sep, f"p{track_id:0>3}.h5"),
            'vd': os.path.join(hdf5s_dir_vd, f"p{track_id:0>3}.h5"),
        }
        
        return positions, hdf5_paths, pitch_offset, text

    def _get_waveform(self, hf, positions, datatype, index):
        sr = hf.attrs['sample_rate']
        if self.tatum_based:
            samples = librosa.time_to_samples(hf['tatum_time'][positions], sr=sr)
        else:
            samples = librosa.frames_to_samples(positions, hop_length=hf.attrs['hop_length'])

        pitch_shift = list(range(-12, 13))[index % 25] if self.configs['pitch_shift'] else 0
        # if len(waveform) < samples[1] - samples[0]:
        #     waveform = np.pad(waveform, (0, samples[1] - samples[0]))
        if pitch_shift == 0:
            return hf['waveform'][samples[0]: samples[1]]
        else:
            return hf[f"waveform_shifted_{pitch_shift}"][samples[0]: samples[1]]
        
    def _get_data(self, hf, positions, pitch_offset, index):
        sr = hf.attrs['sample_rate']
        if self.tatum_based:
            samples = librosa.time_to_samples(hf['tatum_time'][positions], sr=sr)
            frames = librosa.time_to_frames(hf['tatum_time'][positions], sr=sr, hop_length=self.hop_length)
        else:
            samples = librosa.frames_to_samples(positions, hop_length=hf.attrs['hop_length'])
            frames = librosa.samples_to_frames(samples, hop_length=self.hop_length)
        pitch_shift = list(range(-12, 13))[index % 25] if self.configs['pitch_shift'] else 0
        # print(f"pitch_shift={pitch_shift}")
        
        if self.tatum_based:
            tatum_ids = positions
            # Tatums
            tatum_time = hf['tatum_time'][tatum_ids[0]: tatum_ids[1] + 1]
            tatum_time -= tatum_time[0]
            
            # Pitch and onset
            pitches = hf['pitch_tatums'][tatum_ids[0]: tatum_ids[1]]
            # print(pitches)
            pitches[pitches != 128] += pitch_shift
            pitches = np.clip(pitches, 0, 128)
            # print(pitches)
            pitches = self._pitch_to_zeroone(pitches)
            onsets = hf['onset_tatums'][tatum_ids[0]: tatum_ids[1]]
        else:
            tatum_time = 0
            midi_notes = hf['midi_notes'][:]
            pitches, onsets = self._create_frame_based_targets(midi_notes, sr, self.hop_length, frames, pitch_shift=pitch_shift)

        pitches = torch.tensor(pitches, dtype=torch.float)
        onsets = torch.tensor(onsets, dtype=torch.float)

        return pitches, onsets, sr, tatum_time

class DALIDataset(MyDataset):
    
    def __init__(self, indexes_path, hdf5s_dir, 
        **args,
        ):
        # *** self.configs['pitch_shift']
        # 
        super().__init__(indexes_path, hdf5s_dir, **args)
            
    def _get_index(self, i):
        dali_id = self.indexes[i]['dali_id']
        if self.tatum_based:
            positions = self.indexes[i]['tatum_ids']
        else:
            positions = self.indexes[i]['frames']

        if self.codec_type == "with_pitch":
                text = self.indexes[i]['text_with_pitch'] 
        else:
            text = self.indexes[i]['text']
        pitch_offset = self.indexes[i]['offset']
        hdf5_paths = {
            'mix': os.path.join(self.hdf5s_dir, f"{dali_id}.h5"),
            'separated': os.path.join(self.hdf5s_dir, f"{dali_id}.h5"),
            'vd': os.path.join(self.hdf5s_dir, f"{dali_id}.h5"),
        }
        
        return positions, hdf5_paths, pitch_offset, text

    def _get_waveform(self, hf, positions, datatype, index=0):
        key_list = {
            "mix": "waveform",
            "separated": "waveform_separated",
        }
        sr = hf.attrs['sample_rate']
        if self.tatum_based:
            samples = librosa.time_to_samples(hf['tatum_time'][positions], sr=sr)
        else:
            samples = librosa.frames_to_samples(positions, hop_length=hf.attrs['hop_length'])
        
        # if len(waveform) < samples[1] - samples[0]:
        #     waveform = np.pad(waveform, (0, samples[1] - samples[0]))
        return hf[key_list[datatype]][samples[0]: samples[1]]
                
    def _get_data(self, hf, positions, pitch_offset, index=0):
        sr = hf.attrs['sample_rate']
        if self.tatum_based:
            samples = librosa.time_to_samples(hf['tatum_time'][positions], sr=sr)
            frames = librosa.time_to_frames(hf['tatum_time'][positions], sr=sr, hop_length=self.hop_length)
        else:
            samples = librosa.frames_to_samples(positions, hop_length=hf.attrs['hop_length'])
            frames = librosa.samples_to_frames(samples, hop_length=self.hop_length)
        
        if self.tatum_based:
            tatum_ids = positions
            # Tatums
            tatum_time = hf['tatum_time'][tatum_ids[0]: tatum_ids[1] + 1]
            tatum_time -= tatum_time[0]
            
            # Pitch and onset
            pitches = hf['pitch_tatums'][tatum_ids[0]: tatum_ids[1]]
            pitches[pitches != 128] += pitch_offset
            pitches = np.clip(pitches, 0, 128)
            pitches = self._pitch_to_zeroone(pitches)
            onsets = hf['onset_tatums'][tatum_ids[0]: tatum_ids[1]]
        else:
            tatum_time = 0
            midi_notes = hf['annot_notes'][:]
            pitches, onsets = self._create_frame_based_targets(midi_notes, sr, self.hop_length, frames, pitch_offset=pitch_offset)

        pitches = torch.tensor(pitches, dtype=torch.float)
        onsets = torch.tensor(onsets, dtype=torch.float)

        return pitches, onsets, sr, tatum_time


class MyLibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, urls=["train-clean-360"], codec_type="word", channels=1, **args):
        super().__init__()
        self.LibriSpeechDataset = torch.utils.data.ConcatDataset([
            torchaudio.datasets.LIBRISPEECH(
                root=root,
                url=url,
                download=True,
            ) for url in urls
        ])
        self.codec_type = codec_type
        self.codec = Mycodec(self.codec_type)
        self.transform = transform
        self.channels=channels

    def __len__(self):
        return len(self.LibriSpeechDataset)

    def __getitem__(self, i):
        # waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[self.indexes[i]['index']]
        waveform, sr, text, _ , _ , _ = self.LibriSpeechDataset[i]
        text = text.lower()
        
        waveform = waveform.squeeze().numpy()
        input_features = torch.tensor(self.transform(waveform, sr), dtype=torch.float)
        if self.channels > 1:
            input_features = input_features.unsqueeze(0).repeat_interleave(self.channels, dim=0)
        
        if self.codec_type == "phoneme":
            text = self.codec.phonemize(text)
        text = self.codec.encode(text)
        
        n_frames = input_features.shape[-1]
        pitches = torch.zeros(129, n_frames)
        onsets = torch.zeros(n_frames)
        tatum_frames = [0, n_frames]

        return (
                input_features, input_features, pitches, onsets,
                torch.tensor(tatum_frames, dtype=torch.long),
                torch.tensor(text, dtype=torch.int),
                )
