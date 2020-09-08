import shutil
import torch

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def up_date(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def min_max_normalize(tensor):
  tensor_min = torch.min(tensor)
  tensor = tensor - tensor_min
  tensor_max = torch.max(tensor)
  tensor = tensor / tensor_max
  tensor = (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))
  tensor = torch.round(tensor)
  return tensor


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
