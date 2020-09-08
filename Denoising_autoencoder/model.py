import torch.nn.functional as F

from torch import nn


class ModelSq(nn.Module):
  """
  Class for square images (300 x 300).
  """
  def __init__(self):
    super().__init__()
    # encoder layers
    self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
    self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
    self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
        
    # decoder layers
    self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)  
    self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
    self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
    self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
    self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    self.init_conv2d()
    self.init_convtr2d()

  def init_conv2d(self):
    """
    Initialize convolution parameters.
    """
    for c in self.children():
        if isinstance(c, nn.Conv2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.)

  def init_convtr2d(self):
    """
    Initialize convolution parameters.
    """
    for c in self.children():
        if isinstance(c, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.)     

  def forward(self, x):
    # encode
    x = F.relu(self.enc1(x))
    x = self.pool(x)
    x = F.relu(self.enc2(x))
    x = self.pool(x)
    x = F.relu(self.enc3(x))
    x = self.pool(x)
    x = F.relu(self.enc4(x))
    x = self.pool(x) # the latent space representation
        
    # decode
    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = torch.sigmoid(self.out(x))
    
    return x

class ModelRec(nn.Module):
  """
  Class for rectangle images (300 x 400).
  """
  def __init__(self):
    super().__init__()
    # encoder layers
    self.enc1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.enc2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.enc3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.enc4 = nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
    # decoder layers
    self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), output_padding=(0, 1), dilation=(1, 1))  
    self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(0, 1), output_padding=(0, 1), dilation=(1, 1))
    self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), dilation=(1, 1))
    self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), dilation=(1, 1))
    self.out = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    self.init_conv2d()
    self.init_convtr2d()

  def init_conv2d(self):
    """
    Initialize convolution parameters.
    """
    for c in self.children():
        if isinstance(c, nn.Conv2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.)

  def init_convtr2d(self):
    """
    Initialize convolution parameters.
    """
    for c in self.children():
        if isinstance(c, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(c.weight)
            nn.init.constant_(c.bias, 0.)     

  def forward(self, x):
    # encode
    x = F.relu(self.enc1(x))
    x = self.pool(x)
    x = F.relu(self.enc2(x))
    x = self.pool(x)
    x = F.relu(self.enc3(x))
    x = self.pool(x)
    x = F.relu(self.enc4(x))
    x = self.pool(x) # the latent space representation
        
    # decode
    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = torch.sigmoid(self.out(x))
    
    return x
