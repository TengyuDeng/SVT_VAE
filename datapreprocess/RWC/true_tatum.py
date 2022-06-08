import re, pickle, os
import numpy as np
from tqdm import tqdm

tatum_rate = 4

if __name__ == '__main__':
    pkl_dir = "/n/work1/deng/data/RWC/P/true_tatum/"
    os.makedirs(pkl_dir, exist_ok=True)
    for track_id in tqdm(range(1, 101)):
        with open(f"/n/work1/deng/data/RWC/P/beat/RM-P{track_id:0>3}.BEAT.TXT") as f:
            beats = f.read()
        beats = beats.split("\n")
        line = re.split(r'[\t ]', beats[0])
        if line[0] != line[1]:
            flag = 1
        else:
            flag = 0
        beats_time = []
        for line in beats[:-1]:
            line = re.split(r'[\t ]', line)
            beats_time.append(int(line[0]))

        if flag:
            beats_time.append(int(line[1]))
        
        beats_time = np.array(beats_time) / 100
        tatum_time = np.interp(
        np.linspace(1, len(beats_time), (len(beats_time) - 1) * tatum_rate + 1),
        xp = np.linspace(1, len(beats_time), len(beats_time)),
        fp = beats_time,
        )
        pkl_path = os.path.join(pkl_dir, f"p{track_id:0>3}.pkl")
        pickle.dump(tatum_time, open(pkl_path, 'wb'))