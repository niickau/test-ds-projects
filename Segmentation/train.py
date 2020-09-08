import torch
import shutil
import torchvision

import torch.optim as optim
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import AverageMeter, to_np, im_show, save_checkpoint
from dataset import CarvanaDataset
from model import UNETVGG
from loss import BCELoss2d
from PIL import Image


transform_target = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()])
transform_image = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor()])
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = CARVANA(root='/content',
                        mode="train",
                        transform_image=transform_image, transform_target=transform_target)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           shuffle=True)

def train(train_loader, model, optimizer, criterion, num_epochs):
  
  best_loss = 1

  for epoch in range(num_epochs):

    train_loss = AverageMeter()

    model.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        losses.up_date(loss, images.size(0))

        loss.backward()
        optimizer.step()

    is_best = val_loss.avg < best_loss
    best_loss = min(val_loss.avg, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': experiment,
        'state_dict': model.state_dict(),
        'best_prec1': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    
    print('Epoch {} of {}, Train_loss: {:.3f}'.format(epoch + 1, num_epochs, train_loss.avg))


def main():
  torch.cuda.empty_cache()

  model = UNETVGG().cuda()
  criterion = BCELoss2d().cuda()
  optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
  experiment = "Carvana"
  num_epochs = 200

  #checkpoint = torch.load('/content/checkpoint.pth.tar')
  #model.load_state_dict(checkpoint['state_dict'])

  train(train_loader, model, optimizer, criterion, num_epochs)


if __name__ == '__main__':
    main()


  
    
