"""CIFAR-10 scalar energy and input-gradient Langevin dynamics."""
import torch
from torch import nn

class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,16,5,2,2), nn.SiLU(),
            nn.Conv2d(16,32,3,2,1), nn.SiLU(),
            nn.Conv2d(32,64,3,2,1), nn.SiLU(),
            nn.Conv2d(64,64,3,2,1), nn.SiLU(),
            nn.Flatten(), nn.Linear(64*2*2,64), nn.SiLU(), nn.Linear(64,1))
    def forward(self,x):
        return self.layers(x).flatten()

def langevin_sample(model, initial, steps=60, step_size=10.0, noise_std=0.005,
                    rng=None, grad_clip=0.03):
    # autograd.grad returns dE/dx without accumulating model parameter .grad.
    # Do not use inference_mode/no_grad around the whole EBM sampling routine.
    images = initial.detach().clone()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn(images.shape, generator=rng, device=images.device)
            images = (images + noise_std * noise).clamp(-1,1)
        with torch.enable_grad():
            images.requires_grad_(True)
            grad, = torch.autograd.grad(model(images).sum(), images)
        with torch.no_grad():
            images = (images - step_size * grad.clamp(-grad_clip,grad_clip)).clamp(-1,1)
    return images.detach()
