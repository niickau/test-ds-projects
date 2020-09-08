import torch

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import min_max_normalize
from dataset import DAEDataset
from model import ModelRec

transform = transforms.Compose([transforms.Resize((300, 400)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda tensor: min_max_normalize(tensor))])
test_dataset = DAEDataset(root, mode='test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


def test(test_loader, model):
  model.eval()
  for input in test_loader:
    input = input.cuda()
    output = model(input)
    output = output.view(1, 1, 300, 400).cpu().data
    torchvision.utils.save_image(output, 'denoised_test_reconstruction.png')


def main():
  model = ModelDAEConvNew().cuda()
  checkpoint = torch.load('/content/checkpoint.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])

  test(test_loader, model)


if __name__ == '__main__':
    main()
