import torch
import torchvision
from torchvision import datasets, models, transforms
import os

data_dir = '/Users/dauinh/Desktop/COM392/catvsdog/data'

class Dataset:
  def __init__(self):
    self.dir = '/Users/dauinh/Desktop/COM392/catvsdog/data'

  def get_dataset(self):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                          data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                          shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, class_names, dataset_sizes