import shutil
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

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


def to_np(x):
    return x.data.cpu().numpy()


def im_show(img_list):
    
    to_PIL = transforms.ToPILImage()
    if len(img_list) > 9:
        raise Exception('len(img_list) must be smaller than 10')

    for idx, img in enumerate(img_list):
        img = np.array(to_PIL(img))
        plt.subplot(idx + 1)
        fig = plt.imshow(img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
