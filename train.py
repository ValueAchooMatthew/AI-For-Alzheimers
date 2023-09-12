import os

import glob

import numpy as np

import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

###NOTE
# It appears that freesurfers and pups are analytics of the scans
# Include most ad dems regardless of other conditions (?)
# Duplicate data to balance dataset (?)
# max pooling is hella important since the number of input channels is so high


###TODO
# Learn how conv3d
# Find out how to map all nifti's of a given patient to a single label
# Learn how to incorperate labels into this mess
# Learn how to extract all from a folder
# Learn how to standardize image shapes
# Learn difference and pick between nifti-1 and nifti-2 files
# Understand the shape of nifti's
# Create training loop
# die

os.chdir("C:\\Coding\\Pytorch Lessons\\Miscellaneous\\Example NIFTI\\NIFTI's")

images = []

for file in os.listdir(
    "C:\\Coding\\Pytorch Lessons\\Miscellaneous\\Example NIFTI\\NIFTI's"
):
    images.append(nib.load(file).get_fdata())

for image in images:
    image = torch.tensor(np.array(image), dtype=torch.float32)

    print(image.shape)

print(os.system("fslnvols %s" % (image)))

print(image.shape)

image2 = torch.tensor(np.array(image), dtype=torch.float32)

print(image.type())

image = image.unsqueeze(0)

image2 = image.unsqueeze(0)

print(image.shape)

# assuming 256 x 256 is height and width, while 36 is the input channels
image = image.permute(0, 3, 1, 2)

print(image.shape)

print(image + image2)

# [(1) x 256 x 256 x 256]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Out_channels is the number of filters
        # out_channels are also called feature maps
        # The number of in channel in the first convolutional layer depends on the number of colour channels present in the images
        self.conv1 = nn.Conv3d(in_channels=36, out_channels=64, kernel_size=(25, 25))
        # size [1 x 48 x 232 x 232]
        # after pooling [1 x 48 x 112]
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(25, 25))
        # size [1 x 64 x 88 x 88]
        # after pooling [1 x 64 x 40 x 40]

        # fc stands for fully connected
        # out features depends on the number of classes present in the training set
        self.fc1 = nn.Linear(in_features=128 * 40 * 40, out_features=10000)
        self.fc2 = nn.Linear(in_features=10000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=250)
        self.fc4 = nn.Linear(in_features=250, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=2)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=10, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=10, stride=2)

        t = t.reshape(-1, 128 * 40 * 40)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        # must flatten before passing to linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) hidden linear layer
        t = self.fc3(t)
        t = F.relu(t)

        # (7) hidden linear layer
        t = self.fc4(t)
        # Try sigmoid on output for better training https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid
        t = F.sigmoid(t)

        # (8) output layer
        t = self.out(t)

        return t


network = Network()

pred = network(image)

print(pred)

print(F.softmax(pred, dim=1))
