"""Validation-led training; the official test set is only read on explicit request."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from app.cnn import AssignmentCNN, CLASSES, TRANSFORM, PREPROCESSING


def stratified_split(targets, seed=42, per_class=500):
    """Split official training examples only, independently of training RNG."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(targets)
    training, validation = [], []
    for label in range(len(CLASSES)):
        indices = torch.where(targets == label)[0]
        if not 0 < per_class < len(indices):
            raise ValueError('Each class must retain training and validation examples')
        indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
        validation.extend(indices[:per_class])
        training.extend(indices[per_class:])
    return training, validation


def training_transform(augmentation=False):
    if not augmentation:
        return TRANSFORM
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        *TRANSFORM.transforms,
    ])


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


def train_epoch(model, loader, device, optimizer, epoch, epochs):
    """Instructor's CE/Adam/backprop steps, with sample-weighted loss logging."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = correct = 0
    loss_sum = 0.0
    logs = []
    progress = tqdm(loader, ncols=120, desc=f'Epoch {epoch}/{epochs}',
                    disable=not sys.stderr.isatty())
    for batch_number, (inputs, labels) in enumerate(progress):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        predicted = outputs.detach().argmax(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_sum += loss.item() * labels.size(0)
        if (batch_number + 1) % 100 == 0:
            progress.set_postfix(accuracy=f'{correct/total:.3f}', loss=f'{loss_sum/total:.4f}')
            logs.append({'epoch': epoch - 1 + (batch_number + 1) / len(loader),
                         'train_loss': loss_sum / total, 'train_accuracy': correct / total})
    logs.append({'epoch': epoch, 'train_loss': loss_sum / total, 'train_accuracy': correct / total})
    return {'loss': loss_sum / total, 'accuracy': correct / total, 'count': total}, logs


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def rng_state():
    np_state = np.random.get_state()
    state = {'python': random.getstate(), 'torch': torch.get_rng_state(),
             'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]]}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    np_state = state['numpy']
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def inference_checkpoint(model, epoch, config):
    return {'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'classes': list(CLASSES), 'image_size': 64, 'epoch': epoch,
            'seed': config['seed'], 'preprocessing': PREPROCESSING, 'config': config}


def train_model(model, train_loader, validation_loader, device, optimizer, scheduler,
                config, output, epochs, resume=None):
    """Save best validation model separately from complete epoch-boundary resume state."""
    history, datalogs = [], []
    best_key = (-1.0, float('-inf'))
    best_checkpoint = None
    start = 0
    if resume is not None:
        if resume['config'] != config:
            raise ValueError('Resume configuration differs; use the original training settings')
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(resume['scheduler'])
        history, datalogs = resume['history'], resume['datalogs']
        best_checkpoint = resume['best_checkpoint']
        best_key = tuple(resume['best_key'])
        start = resume['epoch']
        restore_rng(resume['rng'])
    if start >= epochs:
        raise ValueError('epochs must exceed completed epochs for continuation')
    for epoch in range(start + 1, epochs + 1):
        learning_rate = optimizer.param_groups[0]['lr']
        training, logs = train_epoch(model, train_loader, device, optimizer, epoch, epochs)
        validation = evaluate(model, validation_loader, device)
        key = (validation['accuracy'], -validation['loss'])
        if key > best_key:
            best_key = key
            best_checkpoint = inference_checkpoint(model, epoch, config)
        if scheduler is not None:
            scheduler.step(validation['loss'])
        record = {'epoch': epoch, 'learning_rate': learning_rate,
                  'training': training, 'validation': validation,
                  'best_epoch': best_checkpoint['epoch']}
        history.append(record)
        datalogs.extend(logs)
        atomic_save(best_checkpoint, output)
        state = inference_checkpoint(model, epoch, config)
        state.update(optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict() if scheduler else None,
                     history=history, datalogs=datalogs, rng=rng_state(),
                     best_checkpoint=best_checkpoint, best_key=list(best_key))
        atomic_save(state, output.with_suffix('.last.pt'))
        report = {'config': config, 'epochs': epoch, 'training_examples': len(train_loader.dataset),
                  'validation_examples': len(validation_loader.dataset),
                  'checkpoint_selection': 'highest validation accuracy; lower validation loss breaks ties',
                  'checkpoint_epoch': best_checkpoint['epoch'],
                  'best_validation_accuracy': best_key[0], 'history': history,
                  'test_evaluated': False}
        output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
        output.with_suffix('.training-log.json').write_text(json.dumps(datalogs, indent=2))
        print(json.dumps(record), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20, help='Total epoch target, including resumed epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='models/cifar10_cnn.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--resume', type=Path, help='Full .last.pt checkpoint, not inference weights')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate existing selected weights without training')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs/batch-size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.evaluate_test:
        from app.cnn import load_classifier
        model = load_classifier(output).to(device)
        dataset = CIFAR10(args.data_dir, train=False, download=True, transform=TRANSFORM)
        result = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), device)
        report = json.loads(output.with_suffix('.metrics.json').read_text())
        report.update(test=result, test_evaluated=True,
                      checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                      test_note='Known benchmark: previously inspected in earlier experiments; not a new blind holdout')
        output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'test': result}), flush=True)
        return
    dataset = CIFAR10(args.data_dir, train=True, download=True,
                      transform=training_transform(args.augmentation))
    deterministic_dataset = CIFAR10(args.data_dir, train=True, download=False, transform=TRANSFORM)
    training_indices, validation_indices = stratified_split(dataset.targets, args.split_seed)
    split = {'seed': args.split_seed, 'training_indices': training_indices,
             'validation_indices': validation_indices, 'source': 'official CIFAR-10 training split only'}
    split_json = json.dumps(split, separators=(',', ':'))
    split_hash = hashlib.sha256(split_json.encode()).hexdigest()
    output.with_suffix('.split.json').write_text(split_json)
    train_loader = DataLoader(Subset(dataset, training_indices), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(deterministic_dataset, validation_indices), batch_size=128)
    config = {'seed': args.seed, 'split_seed': args.split_seed, 'split_sha256': split_hash,
              'batch_size': args.batch_size, 'device': str(device),
              'learning_rate': 0.0005, 'optimizer': 'Adam', 'weight_decay': 0.0,
              'augmentation': args.augmentation, 'scheduler': 'none',
              'scheduler_settings': None,
              'preprocessing': PREPROCESSING,
              'training_loss_aggregation': 'sample-weighted mean', 'threads': 4}
    model = AssignmentCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0)
    scheduler = None
    resume = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    train_model(model, train_loader, validation_loader, device, optimizer, scheduler,
                config, output, args.epochs, resume)
    print('Finished training. Best validation weights saved; test set was not evaluated.', flush=True)


if __name__ == '__main__':
    main()
