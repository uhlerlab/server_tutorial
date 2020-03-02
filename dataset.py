import trainer
from PIL import Image
from torchvision import datasets, transforms
import torch
import os
import torchvision.transforms as transforms
import visdom

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
vis = visdom.Visdom('http://127.0.0.1')
vis.close(env='main')
vis.close(env='data')


def make_dataset():
    files = os.listdir('./data')
    files = ['./data/' + fname for fname in files]
    frames = []
    targets = []

    transform = transforms.Compose(
        [transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
         transforms.ToTensor()])

    for fname in files:
        img = Image.open(fname)
        img = transform(img)
        frames.append(img)
        targets.append(img)


    frames, targets = frames[:1], targets[:1]
    for f in frames:
        vis.image(f, env='data')
    return frames, targets

    
def make_test_dataset():
    files = os.listdir('./data')
    files = ['./data/' + fname for fname in files]
    frames = []
    targets = []

    transform = transforms.Compose(
        [transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
         transforms.ToTensor()])

    for fname in files:
        img = Image.open(fname)
        img = transform(img)
        frames.append(img)
        targets.append(img)

    for f in frames:
        vis.image(f, env='data')
    return frames, targets




    
