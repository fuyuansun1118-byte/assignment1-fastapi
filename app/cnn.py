"""Assignment 2 CNN and shared CIFAR-10 preprocessing."""
import torch
from torch import nn
from torchvision import transforms

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PREPROCESSING = 'rgb_resize64_to_tensor_no_normalization'
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)), transforms.ToTensor(),
])

class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16 * 16, 100),
                                        nn.ReLU(), nn.Linear(100, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


def load_classifier(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != list(CLASSES) or checkpoint['image_size'] != 64:
        raise ValueError('Checkpoint class order or image size does not match the API.')
    if checkpoint.get('preprocessing') != PREPROCESSING:
        raise ValueError('Checkpoint preprocessing does not match the API. Retrain with train.py.')
    model = AssignmentCNN()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
