"""CIFAR-10 native 32x32 RGB tensors, without test-set tuning."""
import torch
from torchvision.datasets import CIFAR10

def load_cifar_tensor(root="data",train=True):
    dataset=CIFAR10(root,train=train,download=True)
    # Keep uint8 until each batch is selected: about 154 MB for the full training set.
    return torch.from_numpy(dataset.data.copy()).permute(0,3,1,2).contiguous()

def image_batches(images,batch_size,rng):
    indices=torch.randperm(len(images),generator=rng)
    for start in range(0,len(images),batch_size):
        yield images[indices[start:start+batch_size]].float()/127.5-1
