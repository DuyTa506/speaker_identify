import os
import shutil
from pathlib import Path
import numpy as np

def assert_out_dir_exists(root, label):
    dir_ = os.path.join(root, label)
    
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        print(f'Created directory {dir_}')
    else:
        print(f'Directory {dir_} already exists')

    return dir_

def train_test_split(root, test_size=0.05):
    # Define paths for train and test directories
    train_dir = f'{root}_train'
    test_dir = f'{root}_test'

    # Create train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process each label (subdirectory)
    for label in os.listdir(root):
        label_path = os.path.join(root, label)
        
        if not os.path.isdir(label_path):
            continue
        
        files_iter = Path(label_path).glob('**/*.npy')
        files_ = [str(f) for f in files_iter]
        files_ = np.array(files_)

        # Create label directories in train and test directories
        assert_out_dir_exists(train_dir, label)
        assert_out_dir_exists(test_dir, label)

        # Shuffle and split the files
        choices = np.random.choice([0, 1], size=files_.shape[0], p=(1 - test_size, test_size))
        train_files = files_[choices == 0]
        test_files = files_[choices == 1]

        # Copy files to the appropriate directories
        for train_sample in train_files:
            dest = os.path.join(train_dir, label, os.path.basename(train_sample))
            print(f'Copying file {train_sample} to {dest}')
            shutil.copyfile(train_sample, dest)

        for test_sample in test_files:
            dest = os.path.join(test_dir, label, os.path.basename(test_sample))
            print(f'Copying file {test_sample} to {dest}')
            shutil.copyfile(test_sample, dest)

        print(f'Done for label: {label}')

    print('All done')

if __name__ == '__main__':
    train_test_split('../fbanks')
