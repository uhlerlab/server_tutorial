import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F

# Abstraction for nonlinearity 
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
        #return x + 4 * torch.sin(x/2)

# Example fully connected network 
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        size = 32
        input_size = 64
        colors = 3
        b = False
        self.input_size = input_size
        self.colors = colors
        module = nn.Sequential(nn.Linear(size, size, bias=b),
                               Nonlinearity())
        num_layers = 9
        self.middle = nn.ModuleList([deepcopy(module) \
                                     for idx in range(num_layers)])
        self.first = nn.Sequential(nn.Linear(input_size*input_size*colors, size,
                                             bias=b),
                                   Nonlinearity(),)
        self.last = nn.Sequential(nn.Linear(size, input_size*input_size*colors,
                                            bias=b))
    
    def forward(self, x):
        o = self.first(x.view(-1, self.colors*self.input_size*self.input_size))
        for idx, m in enumerate(self.middle):
            o = m(o)
        o = self.last(o)
        return o.view(-1, self.colors, self.input_size, self.input_size)
