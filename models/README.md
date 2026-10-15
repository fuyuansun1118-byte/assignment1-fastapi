# Course-configured Assignment 2 CNN

Fresh model, instructor tqdm loop, CrossEntropyLoss, Adam 0.0005, batch size 32,
ten epochs, seed 42, all 50,000 CIFAR-10 training images (15,630 updates).
Assignment-required CNN: RGB 64 x 64, two convolutions, 100-unit hidden FC.
Preprocessing: Resize(64,64) then ToTensor, with no extra normalization.

Fixed final checkpoint: epoch 10. Training accuracy: 76.42%.
Official separate test accuracy: 65.34% (6,534 / 10,000 images).
No test-based checkpoint selection. See cifar10_cnn.metrics.json for all epochs
and cifar10_cnn.training-log.json for the 160 instructor-style log entries.
