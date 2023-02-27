import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
import config

# Load the MNIST dataset and apply transformations
def load_mnist():
    # Load the MNIST dataset
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Number of Training images : {len(train_dataset)}")
    print(f"Number of Testing images : {len(test_dataset)}")
    
    return train_dataset, test_dataset, train_loader, test_loader


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


# Load the Fashion MNIST dataset and apply transformations
def load_fashion_mnist():
    # Load the MNIST dataset
    train_dataset = FashionMNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Number of Training images : {len(train_dataset)}")
    print(f"Number of Testing images : {len(test_dataset)}")
    
    return train_dataset, test_dataset, train_loader, test_loader

# Load the CIFAR 10 dataset and apply transformations
def load_cifar_10():
    # Load the MNIST dataset
    train_dataset = CIFAR10(root='data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Number of Training images : {len(train_dataset)}")
    print(f"Number of Testing images : {len(test_dataset)}")
    
    return train_dataset, test_dataset, train_loader, test_loader


# Load the Custom MNIST dataset and apply transformations
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


class CustomDataset(Dataset):
    def __init__(self, data_params, choice):
        
        self.choice = choice 
        self.samples = torch.Tensor(data_params['inputs']).to(torch.float64)             # Gather: Data Samples
        if(self.choice.lower() == 'train'): 
            self.labels = torch.Tensor(data_params['labels']).to(torch.float64)           # Gather: Data Labels
        
    def __getitem__(self, index):                                                           
        
        if(self.choice.lower() == 'train'): 
            return self.samples[index], self.labels[index]                                  # Return: Next (Sample, Label) 
        else:
            return self.samples[index]                                                     

    def __len__(self):                                                                      
        return len(self.samples)