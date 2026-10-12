"""Train on the official CIFAR-10 training split; evaluate once on test data."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from app.cnn import AssignmentCNN, CLASSES, TRANSFORM


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return {'loss': loss_sum / total, 'accuracy': correct / total, 'count': total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='models/cifar10_cnn.pt')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch-size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = CIFAR10(args.data_dir, train=True, download=True, transform=TRANSFORM)
    train_set, validation_set = random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    model = AssignmentCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        validation = evaluate(model, validation_loader, device)
        record = {'epoch': epoch + 1, 'training_loss': total_loss / len(train_set), 'validation': validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation['accuracy'] > best:
            best = validation['accuracy']
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'classes': list(CLASSES), 'image_size': 64, 'seed': args.seed,
                        'epoch': epoch + 1, 'validation_accuracy': best}, output)
    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_set = CIFAR10(args.data_dir, train=False, download=True, transform=TRANSFORM)
    test_result = evaluate(model, DataLoader(test_set, batch_size=args.batch_size), device)
    report = {'seed': args.seed, 'device': str(device), 'epochs': args.epochs,
              'training_examples': len(train_set), 'validation_examples': len(validation_set),
              'best_epoch': checkpoint['epoch'], 'history': history, 'test': test_result}
    output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'test': test_result}), flush=True)
    # Real held-out examples to make API testing straightforward.
    samples = Path('examples')
    samples.mkdir(exist_ok=True)
    raw_test = CIFAR10(args.data_dir, train=False, download=False)
    for index in range(3):
        image, label = raw_test[index]
        image.save(samples / f'test_{index}_{CLASSES[label]}.png')

if __name__ == '__main__':
    main()
