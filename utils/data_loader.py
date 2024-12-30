import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label

def get_dataloaders(train_transform, test_transform, batch_size):
    train_dataset = CIFAR10Dataset(
        root='./data', 
        train=True, 
        transform=train_transform
    )
    
    test_dataset = CIFAR10Dataset(
        root='./data', 
        train=False, 
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, test_loader 