"""The exact GAN architecture specified in Assignment 3."""
import torch
from torch import nn

class Generator(nn.Module):
    latent_dim = 100
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 7 * 7 * 128),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
    def forward(self, z):
        return self.layers(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1), nn.Sigmoid())
    def forward(self, images):
        return self.layers(images)

class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

def get_model(model_name="GAN"):
    if model_name.upper() != "GAN":
        raise ValueError("This module implements GAN; existing CNN is in app.cnn.")
    return GAN()
