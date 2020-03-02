import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F


# Abstraction for using nonlinearities 
class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        #return F.selu(x)
        #return F.relu(x)
        #return F.leaky_relu(x)
        #return x + torch.sin(10*x)/5
        #return x + torch.sin(x)
        #return x + torch.sin(x) / 2
        #return x + torch.sin(4*x) / 2
        return torch.cos(x) - x
        #return x * F.sigmoid(x)
        #return torch.exp(x)#x**2
        #return x - .1*torch.sin(5*x)

        
# Sample U-Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        size = 64
        k = 2
        b = False
        self.first = nn.Conv2d(3, size, 3, stride=1, padding=1, bias=b)
        self.downsample = nn.Sequential(nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity(),
                                        nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity(),
                                        nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity(),
                                        nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity(),
                                        nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity(),
                                        nn.Conv2d(size, size, 3,
                                                  padding=1, stride=k,
                                                  bias=b),
                                        Nonlinearity())

        self.upsample = nn.Sequential(nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=True),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Upsample(scale_factor=2,
                                                  mode='bilinear',
                                                  align_corners=True),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Conv2d(size, size, 3,
                                                padding=1, stride=1,
                                                bias=b),
                                      Nonlinearity(),
                                      nn.Conv2d(size, 3, 3,
                                                padding=1, stride=1,
                                                bias=b))

    def forward(self, x):
        o = self.first(x)
        o = self.downsample(o)
        o = self.upsample(o)
        return o
