import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
import config

# Load the MNIST dataset and apply transformations
def load_mnist():
    # Load the MNIST dataset
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
    
    # Split the training set into training and validation sets
    train_size = int((1 - config.VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
 
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

def load_mnist_custom_dataset(labels = [1, 7]):
    l1, l2 = labels[0], labels[1]
    
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())

    train_dataset.targets = torch.where(train_dataset.targets == labels[0], torch.tensor(0), train_dataset.targets) 
    train_dataset.targets = torch.where(train_dataset.targets == labels[1], torch.tensor(1), train_dataset.targets) 

    test_dataset.targets = torch.where(test_dataset.targets == labels[0], torch.tensor(0), test_dataset.targets) 
    test_dataset.targets = torch.where(test_dataset.targets == labels[1], torch.tensor(1), test_dataset.targets) 

    train_indices = torch.where((train_dataset.targets == 0) | (train_dataset.targets == 1))[0] 
    test_indices = torch.where((test_dataset.targets == 0) | (test_dataset.targets == 1))[0] 

    train_dataset = Subset(train_dataset, train_indices)  
    test_dataset = Subset(test_dataset, test_indices) 
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Number of Training images with labels {l1} and {l2} : {len(train_dataset)}")
    print(f"Number of Testing images with labels {l1} and {l2}  : {len(test_dataset)}")

    return train_dataset, test_dataset, train_loader, test_loader