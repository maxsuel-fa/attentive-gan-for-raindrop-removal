import os
import random

from math import ceil
from shutil import copy2
from typing import Dict, List, Tuple

def get_paths(dir: str, max_samples: int = 100000) -> List[str]:
    """ Gets the path of all images in the dataset directory.

    Parameters:
        dir (str) -- path for the dataset directory
        max_samples (int) -- maximum number of samples to get from the dataset
                             directory. Default: 100000
    Return (List[str]) -- list where each item is the path for one image
    """
    paths = []
    for dir_name, _, file_names in os.walk(dir):
        for file_name in sorted(file_names):
            paths += [os.path.join(dir_name, file_name)]
    dataset_len = len(paths)
    return paths[:min(dataset_len, max_samples)]

def train_test_split(data_dir: str,
                     g_truth_dir: str,
                     dest_dir: str,
                     ratio: float = 0.75
                     ) -> Dict[str, str]:
    """ TODO """
    directories = dict()

    # creating the train data and ground truth folders
    directories['train_data_dir'] = os.path.join(dest_dir, 'train_data')
    os.makedirs(directories['train_data_dir'])

    directories['train_gtruth_dir'] = os.path.join(dest_dir, 'train_gtruth')
    os.mkdir(directories['train_gtruth_dir'])

    # creating the test data folder
    directories['test_data_dir'] = os.path.join(dest_dir, 'test_data')
    os.mkdir(directories['test_data_dir'])

    data_paths = get_paths(data_dir)
    gtruth_paths = get_paths(g_truth_dir)

    train_data_len = ceil(ratio * len(data_paths))
    train_ids = random.sample(range(len(data_paths)), train_data_len)

    for i in range(len(data_paths)):
        if i in train_ids:
            copy2(data_paths[i], directories['train_data_dir'])
            copy2(gtruth_paths[i], directories['train_gtruth_dir'])
        else:
            copy2(data_paths[i], directories['test_data_dir'])

    return directories
