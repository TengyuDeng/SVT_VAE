import argparse, os, re, shutil
import numpy as np
import DALI as dali_code
import pickle
import h5py
import librosa
from tqdm import tqdm

from utils import read_yaml

def main(args):
    workspace = args.workspace
    dali_dir = args.dataset_dir
    config_yaml = args.config_yaml
    configs = read_yaml(config_yaml)
    offsets = pickle.load(open("/n/work1/deng/data/RWC/annotations/midi_sync_to_wav.pkl", 'rb'))['offsets']
    
    n_fold = configs['n_fold']
    hdf5s_dir = os.path.join(workspace, "hdf5s", "RWC", f"config={os.path.split(config_yaml)[1]}")
    print(f"hdf5s_dir={hdf5s_dir}")

    track_ids = list(range(1, 101))
    for track_id in [3, 23, 64, 66, 77, 81]:
        track_ids.remove(track_id)

    print(f"Creating targets for {len(track_ids)} tracks.")
    
    idx_dir = os.path.join(workspace, "indexes", "RWC", f"config={os.path.split(config_yaml)[1]}")
    print(f"idx_dir={idx_dir}")
    os.makedirs(idx_dir, exist_ok=True)

    random_stat = np.random.RandomState(3756)
    random_stat.shuffle(track_ids)

    fold_idx = np.linspace(0, len(track_ids), n_fold + 1).astype(int)
    for i in range(n_fold):
        print(f"------------Fold{i}------------")
        fold_dir = os.path.join(idx_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        train_ids = track_ids[:fold_idx[i]] + track_ids[fold_idx[i + 1]:]
        test_ids = track_ids[fold_idx[i]:fold_idx[i + 1]]
        train_path = os.path.join(fold_dir, "train_idx.pkl")
        test_path = os.path.join(fold_dir, "test_idx.pkl")
        print("Creating indexes for train set.")
        create_indexes_for_list(train_ids, train_path, hdf5s_dir, configs, offsets)
        print("Creating indexes for test set.")
        create_indexes_for_list(test_ids, test_path, hdf5s_dir, configs, offsets)
    
    shutil.copy(config_yaml, os.path.join(idx_dir, "config.yaml"))

def create_indexes_for_list(track_ids, idx_path, hdf5s_dir, configs, offsets):
    indexes = []
    segment_tatums = configs['segment_tatums']
    sr = configs['sample_rate']
    hop_length = configs['hop_length']

    for track_id in tqdm(track_ids, unit="file"):
        
        hdf5_path = os.path.join(hdf5s_dir, f"p{track_id:0>3}.h5")
        with h5py.File(hdf5_path, 'r') as hf:
            tatum_time = hf['tatum_time'][:]
            try:
                pitch_tatums = hf['pitch_tatums'][:]
            except Exception as e:
                print(f"trackid:{track_id}")
                raise e

        start_tatum = 0
        while True:
            end_tatum = start_tatum + segment_tatums
            if end_tatum >= len(tatum_time):
                if start_tatum < len(tatum_time) - 1:
                    start_tatum = len(tatum_time) - 1 - segment_tatums
                    end_tatum = len(tatum_time) - 1
                else:
                    break
            
            if np.sum(pitch_tatums[start_tatum: end_tatum] == 128) / segment_tatums < 0.9:
                index = {
                'track_id': track_id, 
                'tatum_ids': [start_tatum, end_tatum],
                }
                indexes.append(index)

            start_tatum += segment_tatums
    
    print(f"{len(indexes)} indexes are created in total.")
    print(f"Saving indexes to {idx_path}.")
    pickle.dump(indexes, open(idx_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspace", type=str, default="/n/work1/deng/workspaces/", help="Directory of workspace.")
    parser.add_argument("--dataset_dir", type=str, default="/n/work1/deng/data/DALI", help="Directory of DALI dataset.")
    parser.add_argument("--config_yaml", type=str, default="./datapreprocess/configs/create_hdf5s.yaml", help="Path to configs.")

    args = parser.parse_args()

    main(args)
