import argparse
import os, shutil, re
import time
from tqdm import tqdm

import torch,torchaudio
from openunmix.predict import separate
from openunmix.utils import load_separator

device = "cuda" if torch.cuda.is_available() else "cpu"

def separate_audios(args):
    """
    Separate mp3 files and save vocals as wav files.
    Resample audio files first.

    Args:
        dataset_dir: str
        sample_rate
    """

    dataset_dir = args.dataset_dir

    # Paths
    audios_dir = os.path.join(dataset_dir, "wav")
    separated_dir = os.path.join(dataset_dir, "separated")
    os.makedirs(separated_dir, exist_ok=True)
    
    separator = load_separator(targets=["vocals"], residual=True, device=device)
    separator.freeze()
    model_sr = separator.sample_rate
    print(f"Separator:\n{separator}")
    filenames = sorted(os.listdir(audios_dir))

    for filename in filter(lambda x:re.match(r".*\.wav", x), tqdm(filenames)):

        audio_path = os.path.join(audios_dir, filename)
        separated_path = os.path.join(separated_dir, f"{filename[:filename.rfind('.')]}.wav")
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        with torch.no_grad():
            separated = separate(waveform, sr, separator=separator)['vocals']
        torchaudio.save(separated_path, separated.cpu()[0], model_sr)

        print(f"{audio_path} is separated to vocals and saved as {separated_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/n/work1/deng/data/RWC/P",
        help="Directory of the dataset.",
    )

    # Parse arguments.
    args = parser.parse_args()

    # convert data into wav files.
    separate_audios(args)