import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # input channels, output channels, filter size
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv4 = nn.Conv2d(128, 16, 3)
    self.fc1 = nn.Linear(16 * 12 * 12, 2)
    self.dropout = nn.Dropout(p=0.5)
    self.softmax = nn.Softmax(dim=1)
    # self.fc2 = nn.Linear(1024, 100)
    # self.fc3 = nn.Linear(100, 2)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.softmax(x)
    # x = F.relu(self.fc2(x))
    # x = self.fc3(x)
    return x


class ResBlock(nn.Module):
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
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, input):
    shortcut = self.shortcut(input)
    input = nn.ReLU()(self.bn1(self.conv1(input)))
    input = nn.ReLU()(self.bn2(self.conv2(input)))
    input = input + shortcut
    return nn.ReLU()(input)

class ResNet18(nn.Module):
  def __init__(self, in_channels, resblock, outputs=1000):
    super().__init__()
    self.layer0 = nn.Sequential(
      nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU()
    )

    self.layer1 = nn.Sequential(
      resblock(64, 64, downsample=False),
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
