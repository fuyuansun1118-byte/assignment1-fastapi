"""Protect split integrity, best-model selection and resume equivalence."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import stratified_split, train_model


def test_stratified_split_is_disjoint_reproducible_and_balanced():
    targets = [i for i in range(10) for _ in range(7)]
    training, validation = stratified_split(targets, seed=42, per_class=2)
    assert len(training) == 50 and len(validation) == 20
    assert set(training).isdisjoint(validation)
    assert set(training + validation) == set(range(70))
    assert all(sum(targets[i] == label for i in validation) == 2 for label in range(10))
    assert (training, validation) == stratified_split(targets, seed=42, per_class=2)


def test_resume_matches_uninterrupted_training_and_selects_best(tmp_path):
    def run(output, epochs, resume=None):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = torch.nn.Linear(4, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = None
        generator = torch.Generator().manual_seed(19)
        data = TensorDataset(torch.randn(24, 4, generator=generator), torch.arange(24) % 10)
        loader = DataLoader(data, batch_size=5, shuffle=True)
        validation = DataLoader(data, batch_size=8)
        train_model(model, loader, validation, torch.device('cpu'), optimizer, scheduler,
                    {'seed': 42}, output, epochs, resume)
        return torch.load(output.with_suffix('.last.pt'), weights_only=True)

    full_path, resumed_path = tmp_path / 'full.pt', tmp_path / 'resumed.pt'
    full = run(full_path, 3)
    first = run(resumed_path, 1)
    resumed = run(resumed_path, 3, first)
    assert full['history'] == resumed['history']
    assert full['scheduler'] == resumed['scheduler']
    assert full['optimizer']['param_groups'] == resumed['optimizer']['param_groups']
    for key in full['state_dict']:
        assert torch.equal(full['state_dict'][key], resumed['state_dict'][key])
    best_record = max(full['history'], key=lambda r: (r['validation']['accuracy'], -r['validation']['loss']))
    selected = torch.load(resumed_path, weights_only=True)
    assert selected['epoch'] == best_record['epoch']
    assert selected['epoch'] == full['best_checkpoint']['epoch']
