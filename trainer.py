import neural_model as network
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from copy import deepcopy
import visdom


SIZE = 128
vis = visdom.Visdom('http://127.0.0.1')
vis.close(env='main')
vis.close(env='train')


def train_net(data, labels):

    # Use the following to instantiate a network
    net = network.Net()

    # Use double precision 
    net.double()

    # Put the network on the GPU
    net.cuda()
    # Continue training from a checkpoint by uncommenting:
    #d = torch.load('trained_cnn_model.pth')
    #net.load_state_dict(d['state_dict'])

    # Select your optimization method 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=1e-4)

    # Uncomment this to use a custom initialization
    """
    bound = 5e-2
    for idx, param in enumerate(net.parameters()):
        if idx == 0:
            param.data.fill_(0)
        else:
            init = torch.Tensor(param.size()).uniform_(-bound, bound)
            param.data = init
    #"""

    num_epochs = 1000000

    #Place your data on the GPU
    inputs = Variable(torch.stack(data).double())
    inputs = inputs.cuda()
    targets = Variable(torch.stack(labels).double())
    targets = targets.cuda()

    best_loss = np.float('inf')
    
    for i in range(num_epochs):
        # Take 1 step of GD
        train_loss = train_step(net, inputs, targets, optimizer, iteration=i)

        if i % 100 == 0:
            print(i, train_loss, best_loss)
            vis_output(net, inputs)
            
        # Save the best model if loss is low enough
        if train_loss < best_loss and train_loss < 1e-2:
            best_loss = train_loss
            d = {}
            d['state_dict'] = net.state_dict()
            torch.save(d, 'trained_cnn_model.pth')
            if train_loss < 1e-8:
                break


def train_step(net, inputs, targets, optimizer, iteration=None):
    # Set the network to training mode
    net.train()
    # Zero out all gradients 
    net.zero_grad()
    # Compute the loss (MSE in this case)
    loss = 0.
    outputs = net(inputs)
    if iteration==0:
        print("First output mean: ", outputs[0].mean())
    loss = torch.pow(outputs - targets, 2).mean()
    # Compute backprop updates
    loss.backward()
    # Take a step of GD
    optimizer.step()
    return loss.cpu().data.numpy().item()


def vis_output(net, inputs):
    # Set the network to test mode
    net.eval()
    out = net(inputs)
    out = nn.Upsample(scale_factor=2)(out)
    out = (out - out.min()) / (out.max() - out.min())
    vis.image(out.squeeze(0), env='train')
