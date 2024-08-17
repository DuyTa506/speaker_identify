import os
from pathlib import Path
import argparse
import numpy as np
from data_utils import get_fbanks , train_test_split
np.random.seed(42)

def check_test_size(value):

    
    if not 0 < float(value) < 0.31:
        raise argparse.ArgumentTypeError("Test size must be a float between 0 and 0.3 .")
    return float(value)

def assert_out_dir_exists(output_path, index):
    dir_ = os.path.join(output_path, str(index))

    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print('Created directory {}'.format(dir_))
    else:
        print('Directory {} already exists'.format(dir_))

    return dir_

def main(base_path, output_path, test_size):
    speaker_dirs = [f for f in Path(base_path).iterdir() if f.is_dir()]

    for id , speaker_dir in enumerate(speaker_dirs):
        speaker_id = speaker_dir.name
        print(f'Processing speaker ID: {speaker_id}')

        index_target_dir = assert_out_dir_exists(output_path, id)

        sample_counter = 0
        files_ = list(speaker_dir.glob('**/*.flac'))

        for f in files_:
            fbanks = get_fbanks(str(f))
            if fbanks is None:
                continue
            num_frames = fbanks.shape[0]

            # Sample sets of 64 frames each
            file_sample_counter = 0
            start = 0
            while start < num_frames + 64:
                slice_ = fbanks[start:start + 64]
                if slice_ is not None and slice_.shape[0] == 64:
                    assert slice_.shape[0] == 64
                    assert slice_.shape[1] == 64
                    assert slice_.shape[2] == 1
                    np.save(os.path.join(index_target_dir, f'{sample_counter}.npy'), slice_)

                    file_sample_counter += 1
                    sample_counter += 1

                start = start + 64

            print(f'Done for speaker ID: {speaker_id}, Samples from this file: {file_sample_counter}')

        print(f'Done for speaker ID: {speaker_id}, total number of samples for this ID: {sample_counter}')
        print('')

    print('All done, YAY! Look at the files')
    train_test_split(output_path, test_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract filter banks from audio files.")
    parser.add_argument('input', type=str, help='Input folder containing the audio files.')
    parser.add_argument('out', type=str, help='Output folder to save the extracted features.')
    parser.add_argument('test_size', type=check_test_size, help='Test size.')
    args = parser.parse_args()

    main(args.input, args.out, args.test_size)
