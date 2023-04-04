from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import CenterCrop, Compose, ToTensor

import os
from PIL import Image
from typing import Any, List, Tuple
from math import floor

from .utils import get_paths


class Dataset(TorchDataset):
    """  A handler class that help us to access image data in a convenient way.

    It extends the PyTorch Dataset class in order to facilitate the access
    and paralelization of the load of data.
    """
    def __init__(self, data_dir: str, 
                 g_truth_dir: str,
                 transforms, 
                 max_samples: int = 100000) -> None:
        """ Constructs a new dataset handler class.
        Parameters:
            data_dir (str) -- path to the data directory
            g_truth_dir (str) --  path to the ground truth directory 
            max_samples (int) -- the maximum number of data to get from the
                                 dataset. Default: 100000
        """
        super(Dataset, self).__init__()
        self.data_paths = get_paths(data_dir, max_samples)
        self.g_truth_paths = get_paths(g_truth_dir, max_samples)

        # Gets the height and width of the smallest image to use in the Rezise
        # transformation
        #height, width = self._min_sizes(self.data_paths)

        # Convert such dimensions into the greatest multiple of 16 tha is 
        # small then or equal to them
        #height, width = floor(height / 16) * 16, floor(width / 16) * 16

        #transforms = [CenterCrop((height, width)), ToTensor()]
        transforms += [ToTensor()]
        self.transform = Compose(transforms)

    def _min_sizes(self, paths: List[str]) -> Tuple[int, int]:
        """ Gets the dimensions (height, width) of the smallest image in the 
        dataset.

        Parameters:
            paths (List[str]) -- list where each item is a path for a image in 
                                 the dataset
        Return (Tuple[int, int]) -- the dimensions of the smallest image in the 
                                    dataset
        """
        min_h, min_w = float('inf'), 1
        for img_path in paths:
            img = Image.open(img_path)
            h, w = img.height, img.width
            if h * w < min_h * min_w:
                min_h = h
                min_w = w
            img.close()
        return min_h, min_w

    def __len__(self) -> int:
        """ Gets the size of the dataset 
        Return (int) -- the number of images in the dataset
        """
        return len(self.data_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """ Given the index, gets the image and the respective ground-truth 
        from the dataset 

        Parameters:
            index (int) -- index of the wanted image data
        Return (Tuple[Tensor, Tensor]) -- image data and the respective 
                                          ground-truth
        """
        image_data = Image.open(self.data_paths[index])
        image_data = self.transform(image_data.convert('RGB'))

        ground_truth = Image.open(self.g_truth_paths[index])
        ground_truth = self.transform(ground_truth.convert('RGB'))

        return image_data, ground_truth
