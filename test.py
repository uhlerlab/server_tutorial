import neural_model as network
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from copy import deepcopy
import torch
import visdom
import dataset
from scipy.sparse.linalg import eigs
import options_parser as op

SIZE = 64
COLORS = 3
vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')
vis.close(env='test')


def vis_image(image):
    out = torch.nn.Upsample(scale_factor=2)(deepcopy(image).unsqueeze(0))
    out = (out - out.min()) / (out.max() - out.min())
    vis.image(out.squeeze(0), env='test')
    return out

def iterate(net, image, num_steps=None):
    net.eval()

    o = deepcopy(image)
    x = image
    # Uncomment to add noise to inputs
    """
    l1 = SIZE - SIZE//2
    l2 = SIZE + SIZE//2
    c1 = SIZE - SIZE//2
    c2 = SIZE + SIZE//2
    x[:, l1:l2, c1:c2] = np.random.rand(COLORS, l2 - l1, c2 - c1)
    #"""
    x = torch.from_numpy(x).view(1, COLORS, SIZE, SIZE).double().cuda()
    x.requires_grad = False

    num_iterations = num_steps
    frames = []

    threshold = 1e-17
    diff = np.float("inf")
    count = 0

    vis_image(x[0])

    if num_steps is None:
        num_steps = np.float('inf')
        
    with torch.no_grad():
        while diff > threshold and count < num_steps:
            n_x = net(x)
            count += 1
            diff = torch.mean(torch.pow(n_x - x, 2)).cpu().data.numpy().item()
            if count < 2:
                vis_image(n_x[0])
            del(x)
            x = n_x
            frames.append(deepcopy(x.cpu().data))
            
    original = torch.from_numpy(image).double().unsqueeze(0)
    o = torch.from_numpy(o).double().unsqueeze(0)
    last = frames[-1]

    error =  torch.mean(torch.pow(last - o, 2))

    original = nn.Upsample(scale_factor=2)(original)
    original = (original - original.min()) / (original.max() - original.min())

    last = nn.Upsample(scale_factor=2)(last)
    last = (last - last.min()) / (last.max() - last.min())

    pair = torch.cat([original, last], 0)
    title = 'MSE: ' +  str(error)
    vis.images(pair, nrow=2, opts=dict(title=title),
               env='test')

    return error


def fast_jacobian(net, x, noutputs):
    x = torch.from_numpy(x).double().view(-1, 3,  SIZE, SIZE).cuda()
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(noutputs, 1, 1, 1)
    x.requires_grad_(True)
    y = net(x)
    y = y.view(-1, 3 * SIZE * SIZE)
    y.backward(torch.eye(noutputs).double().cuda())
    return x.grad.data


def main(options):
    seed = options.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    net = network.Net()
    d = torch.load('trained_cnn_model.pth')
    net.load_state_dict(d['state_dict'])
    
    net.double()
    net.cuda()

    train_frames, _ = dataset.make_dataset()
    frames, _  = dataset.make_test_dataset()
    count = 0

    #"""
    for f in train_frames:
        f = deepcopy(f.numpy())
        J = fast_jacobian(net, f, 3 * SIZE * SIZE)
        J = J.view(-1, 3 * SIZE * SIZE)
        J = J.cpu().data.numpy()
        s, _ = eigs(J, k=1, tol=1e-3)
        top = np.abs(s)
        print(top)
        if top < 1:
            count += 1
        del J
    print("Attractors: ", count)
    #"""
    
    #"""
    avg_error = 0
    for f in frames:
        f = deepcopy(f.numpy())
        count += 1
        error = iterate(net, f)
        #print(error)
        avg_error += error
    print(count)
    print("AVERAGE ERROR: ", avg_error/count)
    #"""

    
if __name__ == "__main__":
    options = op.setup_options()
    main(options)    
