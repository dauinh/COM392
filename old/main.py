import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

# Process dataset
train_transform = transforms.Compose([transforms.Resize(255),transforms.RandomResizedCrop(224),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(),
                                transforms.ToTensor()])
test_transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

train_data = datasets.ImageFolder("train", transform=train_transform)
test_data = datasets.ImageFolder("test1", transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

print('Hello World')