import torch
import shutil

import torch.optim as optim
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import AverageMeter, min_max_normalize, save_checkpoint
from dataset import DAEDataset
from model import ModelRec

transform = transforms.Compose([transforms.Resize((300, 400)),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda tensor: min_max_normalize(tensor))])

#working directory
root = '/content'

train_dataset = DAEDataset(root, mode='train', transform=transform)
val_dataset = DAEDataset(root, mode='val', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=True)


def train(train_loader, val_loader, model, optimizer, scheduler, criterion, num_epochs):
  
  best_loss = 1

  for epoch in range(num_epochs):
    scheduler.step()

    train_loss = AverageMeter()
    val_loss = AverageMeter()

    model.train()
    for input, target in train_loader:
      input = input.cuda()
      target = target.cuda()

      optimizer.zero_grad()
      output = model(input)
      
      loss = criterion(output, target)
      train_loss.up_date(loss, input.size(0))

      loss.backward()
      optimizer.step()
    
    print('Epoch {} of {}, Train_loss: {:.3f}'.format(epoch + 1, num_epochs, train_loss.avg))

    model.eval()
    for input, target in val_loader:
      input = input.cuda()
      target = target.cuda()

      output = model(input)
      
      loss = criterion(output, target)
      val_loss.up_date(loss, input.size(0))

    is_best = val_loss.avg < best_loss
    best_loss = min(val_loss.avg, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': experiment,
        'state_dict': model.state_dict(),
        'best_prec1': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)

    
    print('Epoch {} of {}, Val_loss: {:.3f}'.format(epoch + 1, num_epochs, val_loss.avg))


def main():
  torch.cuda.empty_cache()

  model = ModelRec().cuda()
  criterion = nn.BCELoss().cuda()
  optimizer = optim.Adam(model.parameters())
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50,   gamma=0.1)
  experiment = "DAE_Conv"
  num_epochs = 200

  #checkpoint = torch.load('/content/checkpoint.pth.tar')
  #model.load_state_dict(checkpoint['state_dict'])

  train(train_loader, val_loader, model, optimizer, scheduler, criterion, num_epochs)


if __name__ == '__main__':
    main()


  
    
