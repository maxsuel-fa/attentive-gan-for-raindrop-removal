from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

from typing import Tuple


class PairRandomCrop(RandomCrop):
    """ A random crop transformation based on Pytorch RandomCrop to crop a 
    pair of images at the same random location.
    """
    def __init__(self, size: Tuple[int, int]) -> None:
        """ Constructs a new PairRandomCrop transform. 

        Parameters:
            size (Tuple[int, int]) -- the size expected for the output image
        """
        super(PairRandomCrop, self).__init__(size)
        self.odd_call = True

    def forward(self, img):
        """
        Overrides the forward method of RandomCrop.

        Parameters:
            img (PIL Image or Tensor) -- Image to be cropped

        Return (PIL Image or Tensor) -- Cropped image
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        if self.odd_call:
            self.i, self.j, self.h, self.w = self.get_params(img, self.size)
            self.odd_call = not self.odd_call

        return F.crop(img, self.i, self.j, self.h, self.w)
