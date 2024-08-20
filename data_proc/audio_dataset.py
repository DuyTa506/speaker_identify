import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import multiprocessing
import python_speech_features as psf
 


def get_fbanks(audio_file):

    def normalize_frames(signal, epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in signal])

    y, sr = librosa.load(audio_file, sr=16000)
    assert sr == 16000

    trim_len = int(0.25 * sr)
    if y.shape[0] < 1 * sr:
        # if less than 1 seconds, don't use that audio
        return None

    y = y[trim_len:-trim_len]

    # frame width of 25 ms with a stride of 10 ms. This will have an overlap of 15s
    filter_banks, energies = psf.fbank(y, samplerate=sr, nfilt=64, winlen=0.025, winstep=0.01)
    filter_banks = normalize_frames(signal=filter_banks)

    filter_banks = filter_banks.reshape((filter_banks.shape[0], 64, 1))
    return filter_banks



class AudioFBanksDataset(Dataset):
    def __init__(self, root, extensions=('.flac', '.wav', '.mp3', '.m4a')):
        self.audio_files = []
        self.labels = []
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for ext in extensions:
            for file_path in Path(root).rglob(f'*{ext}'):
                self.audio_files.append(file_path)
                label_name = file_path.parent.name
                self.labels.append(self.class_to_idx[label_name])

        self.len_ = len(self.audio_files)
        self.num_classes = len(self.classes)

        # Count instances per class
        bin_counts = np.bincount(self.labels)
        self.label_to_index_range = {}
        start = 0
        for i in range(self.num_classes):
            self.label_to_index_range[i] = (start, start + bin_counts[i])
            start = start + bin_counts[i]

    def __getitem__(self, index):
        audio_path = self.audio_files[index]
        fbanks = get_fbanks(str(audio_path))

        if fbanks is None:
            raise ValueError(f"Failed to extract filter banks from {audio_path}")

        label = self.labels[index]

        num_frames = fbanks.shape[0]
        start = 0

        # Sample 64 frames
        while start < num_frames + 64:
            slice_ = fbanks[start:start + 64]
            if slice_ is not None and slice_.shape[0] == 64:
                assert slice_.shape[1] == 64
                assert slice_.shape[2] == 1

                slice_ = np.moveaxis(slice_, 2, 0)  
                slice_ = torch.from_numpy(slice_).float()

                return slice_, label 
            start += 64

    def __len__(self):
        return self.len_

if __name__ == '__main__':
    use_cuda = False
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}

    data_test = AudioFBanksDataset('../wav_Data')
    test_loader = DataLoader(data_test, batch_size=4, shuffle=True, **kwargs)
    sample, label = next(iter(test_loader))
    print(sample.shape, label)