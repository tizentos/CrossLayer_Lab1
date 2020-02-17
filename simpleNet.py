import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse


# # argument parser
# parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
# parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
# parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
# parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
# parser.add_argument('--enable-cuda',type=bool,default=False,help='Enable cuda')

# args = parser.parse_args()

class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1))
        self.features.add_module("bn1",nn.BatchNorm2d(4))
        self.features.add_module("relu1",nn.ReLU())
        self.features.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv2", nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1))
        self.features.add_module("bn2",nn.BatchNorm2d(16))
        self.features.add_module("relu2",nn.ReLU())
        self.features.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.lin1 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.lin1(out)
        return out

# a = SimpleNet(args)
