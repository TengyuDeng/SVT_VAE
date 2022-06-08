import argparse, os, shutil, re
import numpy as np
import DALI as dali_code
import h5py
import librosa
import pickle
import time

from utils import read_yaml, freq2pitch, wav2spec, wav2mel
from concurrent.futures import ProcessPoolExecutor

def midi_to_tatums(midi_notes, tatum_time):
    pitch_tatums = np.full(len(tatum_time) - 1, 128, dtype=int)
    onset_tatums = np.full(len(tatum_time) - 1, 0, dtype=int)
    i = 0
    j = 0
    pitch_start = 0
    start_time = time.time()
    while True:
        if i >= len(midi_notes) or j >= len(tatum_time) - 1:
            break
    
        while True:
            if i >= len(midi_notes) or j >= len(tatum_time) - 1:
                break
        
            start_diff1 = midi_notes[i][0] - tatum_time[j]
            start_diff2 = tatum_time[j + 1] - midi_notes[i][0]
            start_in_tatum = start_diff1 >= 0 and start_diff2 >= 0

            if start_in_tatum:
                pitch_start = j if start_diff1 < start_diff2 else j + 1
                onset_tatums[pitch_start] = 1
                if midi_notes[i][1] > tatum_time[j + 1]:
                    break
            end_diff1 = midi_notes[i][1] - tatum_time[j]
            end_diff2 = tatum_time[j + 1] - midi_notes[i][1]
            end_in_tatum = end_diff1 >= 0 and end_diff2 >= 0
            if end_in_tatum:
                pitch_end = j if end_diff1 < end_diff2 else j + 1
                pitch_tatums[pitch_start: pitch_end] = midi_notes[i][2]
                onset_tatums[pitch_end] = 1
                i += 1
            else:
                break

        j += 1

    return pitch_tatums, onset_tatums

def main(args):
    workspace = args.workspace
    config_yaml = args.config_yaml
    hdf5s_dir = os.path.join(workspace, "hdf5s", "RWC", f"config={os.path.split(config_yaml)[1]}")
    completed_dir = os.path.join(hdf5s_dir, "completed")
    os.makedirs(completed_dir, exist_ok=True)
    print(f"hdf5s_dir={hdf5s_dir}")
    if not os.path.isdir(hdf5s_dir):
        raise RuntimeError("No such directory!")
    filenames = sorted(os.listdir(hdf5s_dir))
    filenames = list(filter(lambda x:re.match(r'.*\.h5', x), filenames))

    print(f"Creating features for {len(filenames)} hdf5 files.")
    
    params = []

    for filename in filenames:
        hdf5_path = os.path.join(hdf5s_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        params.append((hdf5_path, completed_path))
        
    start_time = time.time()

    with ProcessPoolExecutor() as pool:
        pool.map(create_features, params)

    print("Time used: {:.3f} s".format(time.time() - start_time))

    for filename in sorted(os.listdir(completed_dir)):
        hdf5_path = os.path.join(hdf5s_dir, filename)
        completed_path = os.path.join(completed_dir, filename)
        shutil.move(completed_path, hdf5_path)
    
def create_features(param):
    # start_time = time.time()
    hdf5_path, completed_path = param
    with h5py.File(hdf5_path, 'r') as hf:
        orig_sr = hf.attrs['original_sample_rate']
        sr = hf.attrs['resample_sample_rate']
        hop_length = hf.attrs['hop_length']
        tempo = hf.attrs['tempo']
        waveform = hf['waveform'][:]
        midi_notes = hf['midi_notes'][:]
        tatum_time = hf['tatum_time'][:]
    
    # # Tatums
    # _, beats = librosa.beat.beat_track(y=waveform, sr=orig_sr, hop_length=int(hop_length / sr * orig_sr), start_bpm=tempo, units="time")
    # if len(beats) == 0:
    #     print("Error! Empty waveform!")
    #     return
    # tatum_rate = 4
    # estimated_tatum_time = np.interp(
    #     np.linspace(1, len(beats), (len(beats) - 1) * tatum_rate + 1),
    #     xp = np.linspace(1, len(beats), len(beats)),
    #     fp = beats,
    #     )

    pitch_tatums, onset_tatums = midi_to_tatums(midi_notes, tatum_time)
    
    with h5py.File(hdf5_path, 'w') as hf:
        hf.attrs.create("sample_rate", data=sr, dtype=np.int64)
        hf.attrs.create("hop_length", data=hop_length, dtype=np.int64)
        hf.attrs.create("tempo", data=tempo, dtype=np.float32)
        hf.create_dataset(name="midi_notes", data=midi_notes)
        hf.create_dataset(name="tatum_time", data=tatum_time)
        hf.create_dataset(name="pitch_tatums", data=pitch_tatums, dtype=np.int64)
        hf.create_dataset(name="onset_tatums", data=onset_tatums, dtype=np.int64)

        for i in range(-12, 13):
            if i == 0 : continue
            waveform_shifted = librosa.effects.pitch_shift(waveform, orig_sr, i)
            waveform_shifted = librosa.resample(waveform_shifted ,orig_sr, sr)
            hf.create_dataset(name=f"waveform_shifted_{i}", data=waveform_shifted, dtype=np.float32)
        waveform = librosa.resample(waveform ,orig_sr, sr)
        hf.create_dataset(name="waveform", data=waveform, dtype=np.float32)
    print(f"Successfully create features in {hdf5_path}")
    shutil.move(hdf5_path, completed_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to configs.")

    args = parser.parse_args()

    main(args)
