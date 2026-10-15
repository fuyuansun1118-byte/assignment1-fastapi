# Completed CNN experiments

Both groups completed20 epochs using the same45,000/5,000 stratified split,
seed42, batch32, CrossEntropyLoss and Adam0.0005.

| Group | Best validation accuracy | Epoch | Checkpoint | Metrics |
|---|---:|---:|---|---|
| Baseline | 65.80% | 9 | baseline.pt | baseline.metrics.json |
| Data augmentation | 69.38% | 20 | cifar10_cnn.pt | cifar10_cnn.metrics.json |

The API loads cifar10_cnn.pt. Each group has a .training-log.json and complete
20-epoch history in its metrics file. cifar10_cnn.split.json records their shared
training/validation indices; its SHA-256 is recorded in both configurations.
Scores above are validation accuracy, not test accuracy.
