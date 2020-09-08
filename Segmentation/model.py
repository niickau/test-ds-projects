import torch.nn.functional as F

from torch import nn


def conv3x3(in_, out):
  return nn.Conv2d(in_, out, kernel_size=3, padding=1)


class ConvRelu(nn.Module):

  def __init__(self, in_, out):
    super().__init__()
    self.conv = conv3x3(in_, out)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.conv(x))


class DecoderBlock(nn.Module):

  def __init__(self, in_, mid, out):
    super().__init__()
    self.block = nn.Sequential(
        ConvRelu(in_, mid),
        nn.ConvTranspose2d(mid, out, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.block(x)


class UNETVGG(nn.Module):

  def __init__(self, num_features=32, num_classes=1):
    super().__init__()

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # pretrained model
    self.encoder = torchvision.models.vgg11(pretrained=True).features
    self.relu = self.encoder[1]
    self.conv_1 = self.encoder[0]
    self.conv_2 = self.encoder[3]
    self.conv_3s = self.encoder[6]
    self.conv_3 = self.encoder[8]
    self.conv_4s = self.encoder[11]
    self.conv_4 = self.encoder[13]
    self.conv_5s = self.encoder[16]
    self.conv_5 = self.encoder[18]

    self.center = DecoderBlock(num_features * 8 * 2, num_features * 8 * 2, num_features * 8)

    self.dec_5 = DecoderBlock(num_features * (16 + 8), num_features * 8 * 2, num_features * 8)
    self.dec_4 = DecoderBlock(num_features * (16 + 8), num_features * 8 * 2, num_features * 4)
    self.dec_3 = DecoderBlock(num_features * (8 + 4), num_features * 4 * 2, num_features * 2)
    self.dec_2 = DecoderBlock(num_features * (4 + 2), num_features * 2 * 2, num_features)
    self.dec_1 = ConvRelu(num_features * (2 + 1), num_features)

    self.final = nn.Conv2d(num_features, num_classes, kernel_size=1)


  def forward(self, x):

    #encoder
    down_1 = self.relu(self.conv_1(x))
    down_2 = self.relu(self.conv_2(self.pool(down_1)))
    down_3s = self.relu(self.conv_3s(self.pool(down_2)))
    down_3 = self.relu(self.conv_3(down_3s))
    down_4s = self.relu(self.conv_4s(self.pool(down_3)))
    down_4 = self.relu(self.conv_4(down_4s))
    down_5s = self.relu(self.conv_5s(self.pool(down_4)))
    down_5 = self.relu(self.conv_5(down_5s))

    center = self.center(self.pool(down_5))

    #decoder
    up_5 = self.dec_5(torch.cat([center, down_5], 1))
    up_4 = self.dec_4(torch.cat([up_5, down_4], 1))
    up_3 = self.dec_3(torch.cat([up_4, down_3], 1))
    up_2 = self.dec_2(torch.cat([up_3, down_2], 1))
    up_1 = self.dec_1(torch.cat([up_2, down_1], 1))

    result = self.final(up_1)

    return result
