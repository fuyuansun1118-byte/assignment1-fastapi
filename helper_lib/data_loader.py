"""MNIST training images scaled to match the generator's Tanh output."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loader(data_dir="data", batch_size=128, train=True, seed=42):
    dataset = datasets.MNIST(data_dir, train=train, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))
