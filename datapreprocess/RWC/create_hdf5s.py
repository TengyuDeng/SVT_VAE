import argparse, os, shutil
import numpy as np
import DALI as dali_code
import h5py
from tqdm import tqdm
import librosa
import pickle
import pretty_midi
import time

from utils import read_yaml, freq2pitch, wav2spec, wav2mel

# tempi = [  0, 135, 100, 111,  86, 135, 120, 122, 127,  70, 125,  90, 120, 103,  88, 132, 122,  97, 112, 130, 
#          134,  98, 135, 132, 130, 103, 158, 124, 109, 103, 104, 129, 125, 108,  93, 170, 135, 184, 125,  73,
#          122, 200, 125, 163, 124,  77, 168,  94,  86, 100, 114, 104, 140, 132, 125,  74,  74,  70, 118,  98,
#          148, 121,  81, 126, 100,  76,  76,  88,  92,  62, 104,  70,  76, 144,  94, 108,  70, 120,  75,  92,
#           80,  90,  88, 140, 138,  98,  80,  90, 120, 134, 127, 124, 134,  90,  78, 161, 130,  91, 107,  73,  
#           80, ]


def main(args):
    # HOP LENGTH is fixed in hdf5s !
    workspace = args.workspace
    data_dir = args.dataset_dir
    config_yaml = args.config_yaml
    configs = read_yaml(config_yaml)
    data_type = configs['data_type']
    sr = configs['sample_rate']
    hop_length = configs['hop_length']
    if data_type == "separated":
        wav_dir = os.path.join(data_dir, "separated") # RM-P{track_id:0>3}.wav
    elif data_type == "vd":
        wav_dir = "/n/work1/deng/data/RWC/P/vcl/wav/RWC-MDB-P-2001.VOCAL_DRY/"
    else:
        wav_dir = os.path.join(data_dir, "wav") # RM-P{track_id:0>3}.wav

    midi_dir = os.path.join(data_dir, "smf.sync") # p{track_id:0>3}.mid
    hdf5s_dir = os.path.join(workspace, "hdf5s", "RWC", f"config={os.path.split(config_yaml)[1]}")
    print(f"hdf5s_dir={hdf5s_dir}")
    os.makedirs(hdf5s_dir, exist_ok=True)

    track_ids = list(range(1, 101))
    for track_id in [3, 23, 64, 66, 77, 81]:
        track_ids.remove(track_id)

    print(f"Creating targets for {len(track_ids)} tracks.")
    
    sync_data = pickle.load(open("/n/work1/deng/data/RWC/annotations/midi_sync_to_wav.pkl", 'rb'))
    tempi = pickle.load(open("/n/work1/deng/data/RWC/annotations/tempi.pkl", 'rb'))
    offsets = sync_data['offsets']
    melody_id = sync_data['melody_id']

    for track_id in tqdm(track_ids, unit="file"):
        if track_id in [2, 7, 14, 37, 39, 42, 50, 53, 54, 62, 69, 75, 77, 81, 89]:
            pitch_offset = 0
        elif track_id in [1, 30]:
            pitch_offset = - 2 * 12
        else:
            pitch_offset = - 1 * 12
        midi_data = pretty_midi.PrettyMIDI(os.path.join(midi_dir, f"p{track_id:0>3}.mid"))
        tatum_time = pickle.load(open(f"/n/work1/deng/data/RWC/P/true_tatum/p{track_id:0>3}.pkl", 'rb'))
        if data_type == "vd":
            waveform, orig_sr = librosa.load(os.path.join(wav_dir, f"p{track_id:0>3}vd.wav"), sr=None)
        else:
            waveform, orig_sr = librosa.load(os.path.join(wav_dir, f"RM-P{track_id:0>3}.wav"), sr=None)
        notes = midi_data.instruments[melody_id[track_id]].notes
        tempo = tempi[track_id]
        offset = offsets[track_id]
        midi_notes = []
        for note in notes:
            midi_notes.append([note.start - offset, note.end - offset, note.pitch + pitch_offset])
        midi_notes = np.array(midi_notes)
        hdf5_path = os.path.join(hdf5s_dir, f"p{track_id:0>3}.h5")
        
        # create_target(waveform, orig_sr, notes, tempo, pitch_offset, hdf5_path, configs)
        with h5py.File(hdf5_path, 'w') as hf:
            hf.attrs.create("original_sample_rate", data=orig_sr, dtype=np.int64)
            hf.attrs.create("resample_sample_rate", data=sr, dtype=np.int64)
            hf.attrs.create("hop_length", data=hop_length, dtype=np.int64)
            hf.attrs.create("tempo", data=tempo, dtype=np.float32)
            # Not resampled yet!
            hf.create_dataset(name="waveform", data=waveform)
            hf.create_dataset(name="midi_notes", data=midi_notes)
            hf.create_dataset(name="tatum_time", data=tatum_time)

    shutil.copy(config_yaml, os.path.join(hdf5s_dir, "config.yaml"))

# def create_target(waveform, orig_sr, notes, tempo, pitch_offset, hdf5_path, configs):
#     # start_time = time.time()
#     sr = configs['sample_rate']
#     hop_length = configs['hop_length']

#     # Pitches & Onsets
#     length = int(len(waveform) / orig_sr * sr) // hop_length + 1
#     pitches = np.full(length, 128)
#     onsets = np.full(length, 0)

#     for note in notes:
#         start_time = librosa.time_to_frames(note.start, sr=sr, hop_length=hop_length)
#         end_time = librosa.time_to_frames(note.end, sr=sr, hop_length=hop_length)
#         pitch_id = note.pitch + pitch_offset

#         if end_time < len(pitches) and start_time < end_time:
#             if pitch_id >= 0 and pitch_id < 128:
#                 pitches[start_time: end_time] = pitch_id
#                 onsets[start_time] = 1
#                 onsets[end_time] = 1 # onsets for rest
    
#     with h5py.File(hdf5_path, 'w') as hf:
#         hf.attrs.create("original_sample_rate", data=orig_sr, dtype=np.int64)
#         hf.attrs.create("resample_sample_rate", data=sr, dtype=np.int64)
#         hf.attrs.create("hop_length", data=hop_length, dtype=np.int64)
#         hf.attrs.create("tempo", data=tempo, dtype=np.float32)
#         # Not resampled yet!
#         hf.create_dataset(name="waveform", data=waveform)
#         hf.create_dataset(name="pitch_frames", data=pitches, dtype=np.int64)
#         hf.create_dataset(name="onset_frames", data=onsets, dtype=np.int64)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--dataset_dir", type=str, default="/n/work1/deng/data/RWC/P/", help="Directory of DALI dataset.")
    parser.add_argument("--config_yaml", type=str, default="./datapreprocess/configs/create_hdf5s.yaml", help="Path to configs.")
    parser.add_argument("--data_type", type=str, default="mix", help="data type")
    args = parser.parse_args()

    main(args)
