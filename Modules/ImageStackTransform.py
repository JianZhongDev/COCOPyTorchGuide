"""
FILENAME: ImageStackTransform.py
DESCRIPTION: Define transforms for a stack of images 
@author: Jian Zhong
"""


import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2


# random crop function for image stack
class RandomCrop(v2.RandomCrop):
    def __init__(
            self,
            **args
    ):
        """
        Refer v2.RandomCrop documentation for agrument definition
        """
        super().__init__(**args)

    def forward(self, src_image_stack):
        dst_image_stack = [None for _ in range(len(src_image_stack))]
        
        i, j, h, w = 0, 0, 0, 0

        for i_img in range(len(src_image_stack)):
            cur_src_image = src_image_stack[i_img]

            if self.padding is not None:
                cur_src_image = F.pad(cur_src_image, self.padding, self.fill, self.padding_mode)

            _, height, width = F.get_dimensions(cur_src_image)
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                cur_src_image = F.pad(cur_src_image, padding, self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                cur_src_image = F.pad(cur_src_image, padding, self.fill, self.padding_mode)

            if i_img == 0:
                i, j, h, w = self.get_params(cur_src_image, self.size)

            dst_image_stack[i_img] = F.crop(cur_src_image, i, j, h, w)
        
        return dst_image_stack
    

# random horizontal flip
class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p = 0.5):
        """
        p (float): probability of the image being flipped.
        """
        super().__init__()
        self.p = p
    
    def forward(self,src_image_stack):
        dst_image_stack = src_image_stack
        if torch.rand(1) < self.p:
            dst_image_stack = [None for _ in range(len(src_image_stack))]
            for i_img in range(len(src_image_stack)):
                cur_src_image = src_image_stack[i_img]
                dst_image_stack[i_img] = F.hflip(cur_src_image)

        return dst_image_stack
    

# random vertial flip
class RandomVerticalFlip(torch.nn.Module):

    def __init__(self, p = 0.5):
        """
        p (float): probability of the image being flipped.
        """
        super().__init__()
        self.p = p
    
    def forward(self,src_image_stack):
        dst_image_stack = src_image_stack
        if torch.rand(1) < self.p:
            dst_image_stack = [None for _ in range(len(src_image_stack))]
            for i_img in range(len(src_image_stack)):
                cur_src_image = src_image_stack[i_img]
                dst_image_stack[i_img] = F.vflip(cur_src_image)

        return dst_image_stack