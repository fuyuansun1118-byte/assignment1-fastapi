"""Train on the official CIFAR-10 training split; evaluate once on test data."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm
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
    # Instructor-supplied optimizer and training loop. A fresh model is trained.
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    datalogs = []
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0
        model.train()
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{args.epochs}"
        )
        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()
            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({
                    "avg accuracy": f"{running_correct/running_total:.3f}",
                    "avg loss": f"{running_loss/(batch_number+1):.4f}",
                })
                datalogs.append({
                    "epoch": epoch + batch_number / len(train_loader),
                    "train_loss": running_loss / (batch_number + 1),
                    "train_accuracy": running_correct / running_total,
                })
        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "train_accuracy": running_correct / running_total,
        })
        # The instructor's widget.src/losschart line only updates a notebook plot.
        # Persist the same datalogs below, so this script also works outside Jupyter.
        validation = evaluate(model, validation_loader, device)
        record = {'epoch': epoch + 1, 'training_loss': running_loss / len(train_loader),
                  'training_accuracy': running_correct / running_total, 'validation': validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation['accuracy'] > best:
            best = validation['accuracy']
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'classes': list(CLASSES), 'image_size': 64, 'seed': args.seed,
                        'epoch': epoch + 1, 'validation_accuracy': best}, output)
    print('Finished Training', flush=True)
    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    test_set = CIFAR10(args.data_dir, train=False, download=True, transform=TRANSFORM)
    test_result = evaluate(model, DataLoader(test_set, batch_size=args.batch_size), device)
    report = {'seed': args.seed, 'device': str(device), 'epochs': args.epochs,
              'training_loop': 'instructor-provided tqdm loop', 'learning_rate': 0.0005,
              'optimizer': 'Adam', 'batch_size': args.batch_size,
              'training_loss_aggregation': 'mean of per-batch losses, as in instructor loop',
              'training_examples': len(train_set), 'validation_examples': len(validation_set),
              'best_epoch': checkpoint['epoch'], 'history': history, 'test': test_result}
    output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
    output.with_suffix('.training-log.json').write_text(json.dumps(datalogs, indent=2))
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
