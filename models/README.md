# Instructor-loop CIFAR-10 CNN

Fresh initialization, instructor-provided tqdm training loop, CrossEntropyLoss,
Adam learning rate 0.0005, ten epochs, batch size 128, seed 42.
Exact Assignment 2 CNN architecture and shared 64 x 64 RGB preprocessing.

Best validation checkpoint: epoch 10 (64.16%). Independent official
test accuracy: 64.84% (6,484 / 10,000 images). Final-epoch training
accuracy: 76.62%. See cifar10_cnn.metrics.json for per-epoch measurements and
cifar10_cnn.training-log.json for the instructor's datalogs.
