import torch
import os

from PIL import Image
from os.path import isfile, join
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CarvanaDataset(Dataset):

    def __init__(self, root, mode="train", transform_image=None, transform_target=None):

        self.root = os.path.expanduser(root)
        self.transform_image = transform_image
        self.transform_target = transform_target
        self.mode = mode
        self.data_path, self.labels_path = [], []

        def load_images(path):
            
            images_dir = [join(path, file) for file in os.listdir(path) if isfile(join(path, file))]
            images_dir.sort()

            return images_dir

        if self.mode == "train":
            self.data_path = load_images(self.root + "/train/train")
            self.labels_path = load_images(self.root + "/train_masks/train_masks")
        elif self.mode == "val":
            self.data_path = load_images(self.root + "/val")
            self.labels_path = load_images(self.root + "/val_masks")
        elif self.mode == "test":
            self.data_path = load_images(self.root + "/test")
            self.labels_path = None
        else:
            raise RuntimeError('Invalid subset ' + self.mode + ', it must be one of:'
                                                                 ' \'train\', \'val\' or \'test\'')
            
    def __getitem__(self, index):
        
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index]) if self.mode != 'test' else None
        
        if self.transform_image is not None:
            img = self.transform_image(img)
        if self.transform_target is not None:
            target = self.transform_target(target)
            
        return img, target
    
    def __len__(self):
        return len(self.data_path)
