import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, downsample):
    super().__init__()
    if downsample:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      self.shortcut = nn.Sequential()

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, input):
    shortcut = self.shortcut(input)
    input = nn.ReLU()(self.conv1(input))
    input = nn.ReLU()(self.conv2(input))
    input = input + shortcut
    return nn.ReLU()(input)

class Net(nn.Module):
  def __init__(self, in_channels, resblock, outputs=2):
    super().__init__()
    self.layer0 = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
      nn.ReLU()
    )

    self.layer1 = nn.Sequential(
      resblock(32, 64, downsample=True),
      resblock(64, 64, downsample=False)
    )

    self.layer2 = nn.Sequential(
      resblock(64, 128, downsample=True),
      resblock(128, 128, downsample=False)
    )

    self.layer3 = nn.Sequential(
      resblock(128, 256, downsample=True),
      resblock(256, 256, downsample=False)
    )


    self.layer4 = nn.Sequential(
      resblock(256, 512, downsample=True),
      resblock(512, 512, downsample=False)
    )

    self.gap = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Linear(512, outputs)

  def forward(self, input):
    input = self.layer0(input)
    input = self.layer1(input)
    input = self.layer2(input)
    input = self.layer3(input)
    input = self.layer4(input)
    input = self.gap(input)
    input = torch.flatten(input, 1)
    input = self.fc(input)

    return input
