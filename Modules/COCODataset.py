"""
FILENAME: COCODataset.py
DESCRIPTION: Pytorch COCO dataset using pycocotools 
@author: Jian Zhong
"""

import os
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.io import read_image


## COCO segmentation dataset
class COCOSegDataset(Dataset):

    # helper private method get local image file path according to the coco_img dict structure
    def __getLocalImagePath(self, coco_img):
        local_img_path = os.path.join(self.image_dir_path, coco_img["file_name"])
        return local_img_path

    def __init__(self, 
        annotation_file_path,
        image_dir_path = None,
        category_names = [],
        data_transform = None,
        target_transform = None,
        common_transform = None,
        color_categories = False,
        split_segmentations = False,
        overwrite_local_image = False,
    ):
        """
        annotation_file_path : file path for the annotation file
        image_dir_path : directory where the image store
        category_names : name of selected categoris
        data_transform : transform of source image
        target_transform: transform of target image
        common_transform : transform shared with data and label
        color_categories : indicate whether coloring segments with their id
        split_segmentations : incidate whether split different categories into differenet images
        overwrite_local_image : indicate whether overwriting local image
        """

        self.image_dir_path = image_dir_path
        self.color_categories = color_categories
        self.split_segmentations = split_segmentations

        self.data_transform = data_transform
        self.target_transform = target_transform
        self.common_transform = common_transform 

        # create COCO object
        self.coco_obj = COCO(annotation_file = annotation_file_path)
        
        # create summary category, image, and annotation ids
        self.cat_ids = self.coco_obj.getCatIds(catNms = category_names)
        self.img_ids = self.coco_obj.getImgIds(catIds = self.cat_ids)
        self.ann_ids = self.coco_obj.getAnnIds(imgIds = self.img_ids, catIds = self.cat_ids)

        # create download image to local if necessary
        if self.image_dir_path is None:
            root_dir_path = os.path.split(annotation_file_path)[0]
            self.image_dir_path = os.path.join(root_dir_path, "images")

        # iterate through all the image ids check if image have been downloaded
        download_img_ids = []
        if not overwrite_local_image:
            for cur_img in self.coco_obj.loadImgs(ids = self.img_ids):
                cur_img_local_path = self.__getLocalImagePath(cur_img)
                if not os.path.exists(cur_img_local_path):
                    download_img_ids.append(cur_img["id"])
        else:
            download_img_ids = self.img_ids

        # download image if it hasn't been downloaded
        if not os.path.isdir(self.image_dir_path):
            os.makedirs(self.image_dir_path)
        if(len(download_img_ids) > 0):
            self.coco_obj.download(tarDir=self.image_dir_path, imgIds = download_img_ids)


    def __len__(self):
        if self.split_segmentations:
            return len(self.ann_ids)
        else:
            return len(self.img_ids)
    

    def __getitem__(self, idx):
        
        # get coco imgs and anns
        cur_imgs = None
        cur_anns = None
        if self.split_segmentations:
            cur_ann_ids = [self.ann_ids[idx]]
            cur_anns = self.coco_obj.loadAnns(cur_ann_ids)
            cur_img_ids = [cur_anns[0]["image_id"]]
            cur_imgs = self.coco_obj.loadImgs(cur_img_ids)
        else:
            cur_img_ids = [self.img_ids[idx]]
            cur_imgs = self.coco_obj.loadImgs(cur_img_ids)
            cur_ann_ids = self.coco_obj.getAnnIds(imgIds = cur_img_ids, catIds = self.cat_ids)
            cur_anns = self.coco_obj.loadAnns(cur_ann_ids)
        
        # get image and target tensors
        cur_img = cur_imgs[0]
        cur_data = read_image(self.__getLocalImagePath(cur_img))
        cur_target = torch.zeros(1, cur_img["height"], cur_img["width"], dtype = torch.float)
        for cur_ann in cur_anns:
            cur_mask = self.coco_obj.annToMask(cur_ann)
            cur_fill_val = 1
            if(self.color_categories):
                cur_fill_val = cur_ann["category_id"]
            cur_target[:, cur_mask > 0] = cur_fill_val

        # apply transforms
        if self.data_transform is not None:
            cur_data = self.data_transform(cur_data)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        if self.common_transform is not None:
            cur_data_pkg = [cur_data, cur_target]
            cur_data_pkg = self.common_transform(cur_data_pkg)
            cur_data = cur_data_pkg[0]
            cur_target = cur_data_pkg[1]

        return cur_data, cur_target

    def get_categories(self):
        return dict(self.coco_obj.dataset["categories"])
    




        

