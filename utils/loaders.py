# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.datasets import ImageFolder



class Args(object):
  def __init__(self, size_image=224, batch_size=32, path_to_dir = './data/data_original/Aus/0d'):
    self.size_image = size_image
    self.batch_size = batch_size
    self.root = path_to_dir

class ImageTransform(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), resize=224, centercrop=1200, fill=255):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.CenterCrop(centercrop),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(0, 360), fill=fill),
                transforms.Resize((resize, resize)), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'valid': transforms.Compose([
                transforms.CenterCrop(centercrop),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'no_Normalize' : transforms.Compose([transforms.CenterCrop(centercrop),
                                            transforms.Resize((resize, resize)),
                                            transforms.ToTensor(),
                                            ])
        }

    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)

class MarchantiaDataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0]) #img
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1] #label
        return x, y
    
    def __len__(self):
        return len(self.dataset)

class MaskDataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(rgb_to_grayscale(self.dataset[index][0])) #img
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1] #label
        return x, y
    
    def __len__(self):
        return len(self.dataset)

def get_data(args, full_dataset, Transform, MarchantiaDataset, seed=None):
    targets = np.array(full_dataset.targets)
    dataset_noaug = MarchantiaDataset(full_dataset, Transform.data_transform["valid"])
    dataset_aug = MarchantiaDataset(full_dataset, Transform.data_transform["train"])
    train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), test_size=0.2, stratify=targets, random_state=seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, stratify=targets[train_indices], random_state=seed)

    train_dataset = data.Subset(dataset_aug, indices=train_indices)
    val_dataset = data.Subset(dataset_noaug, indices=val_indices)
    test_dataset = data.Subset(dataset_noaug, indices=test_indices)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dataset_attributes = {'n_train': len(train_dataset), 'n_val': len(val_dataset), 'n_test': len(test_dataset)}
    dataset_indices = {'targets': targets,
                       'idx_train': train_indices, 'idx_val': val_indices, 'idx_test': test_indices}
                          
    return train_loader, val_loader, test_loader, dataset_attributes, dataset_indices


def get_data_rotatedTestset(args, full_dataset, Transform, MarchantiaDataset, seed=None):
    targets = np.array(full_dataset.targets)
    dataset_aug = MarchantiaDataset(full_dataset, Transform.data_transform["train"])
    train_indices, test_indices = train_test_split(np.arange(targets.shape[0]), test_size=0.2, stratify=targets, random_state=seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, stratify=targets[train_indices], random_state=seed)

    train_dataset = data.Subset(dataset_aug, indices=train_indices)
    val_dataset = data.Subset(dataset_aug, indices=val_indices)
    test_dataset = data.Subset(dataset_aug, indices=test_indices)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dataset_attributes = {'n_train': len(train_dataset), 'n_val': len(val_dataset), 'n_test': len(test_dataset)}
    dataset_indices = {'targets': targets,
                       'idx_train': train_indices, 'idx_val': val_indices, 'idx_test': test_indices}
                          
    return train_loader, val_loader, test_loader, dataset_attributes, dataset_indices

def indices_to_loader(model_path, indices_csv, dataset, batch_size):
    indices = pd.read_csv(model_path + indices_csv, header = None)
    indices = np.squeeze(indices.values.astype(int))
    dataset = data.Subset(dataset, indices=indices)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle= False)
    return loader
