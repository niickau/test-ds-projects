import torch

from PIL import Image
from os.path import isfile, join
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class DAEDataset(Dataset):
  def __init__(self, root, mode, transform):
    super().__init__()
    self.root = root
    self.mode = mode
    self.transform = transform
    self.input_path, self.output_path = [], []

    def load_images(path):
      images = [os.path.join(path, file) for file in os.listdir(path)]
      images.sort()
      return images

    if self.mode == 'train':
      self.input_path = load_images(self.root + '/dataset_dae/train/train/before')
      self.output_path = load_images(self.root + '/dataset_dae/train/train/after')
    elif self.mode == 'val':
      self.input_path = load_images(self.root + '/dataset_dae/train/val/before')
      self.output_path = load_images(self.root + '/dataset_dae/train/val/after')
    elif self.mode == 'test':
      self.input_path = load_images(self.root + '/dataset_dae/test')
    else:
      raise RuntimeError('Invalid subset ' + self.mode + ', it must be one of:'
                                                                 ' \'train\', \'val\' or \'test\'')
    
  def __getitem__(self, index):
    img_in = Image.open(self.input_path[index])
    input = self.transform(img_in)
    if self.mode == 'test':
      return input
    else:
      img_out = Image.open(self.output_path[index])
      output = self.transform(img_out)
      return input, output

  def __len__(self):
    return len(self.input_path)
