# Assignment 2 CNN Image Classification

PyTorch implementation of the required CIFAR-10 CNN, integrated into the
Assignment 1 FastAPI application. The completed baseline and data augmentation
experiments use the same training/validation split. Data augmentation achieves
**69.38% validation accuracy**, compared with **65.80%** for the baseline.

## Model and training method

RGB input 64×64; Conv(3,16,3,stride1,padding1), ReLU, MaxPool(2,2);
Conv(16,32,3,stride1,padding1), ReLU, MaxPool(2,2); Flatten8192;
Linear(8192,100), ReLU, Linear(100,10). Total: **825,398 parameters**.
CrossEntropyLoss operates on logits; softmax is used for API probabilities.

The official 50,000 CIFAR-10 training images are stratified into **45,000 training
and 5,000 validation images**, with 500 validation examples per class.
Both experiments use seed 42, split seed 42, batch size 32, Adam learning rate 0.0005,
no weight decay, constant learning rate, and **20 completed epochs** on CPU.
Base preprocessing resizes images to 64×64 and converts pixels to [0,1], without
additional normalization. Only the augmentation group applies RandomCrop(32,
padding4) and RandomHorizontalFlip(p=0.5) before resizing. Validation and API
preprocessing are deterministic. The instructor's forward, loss, backward and
optimizer-update steps are implemented in `train_epoch`; `train_model` evaluates
validation after each epoch and saves the best checkpoint. Highest validation
accuracy selects the epoch; lower validation loss breaks ties.

## Completed validation results

| Experiment | Completed epochs | Best validation accuracy | Best epoch | Weights |
|---|---:|---:|---:|---|
| Baseline | 20 | 65.80% | 9 | models/baseline.pt |
| Random crop and horizontal flip | 20 | 69.38% | 20 | models/cifar10_cnn.pt |

Augmentation improves the best validation score by **3.58 percentage points**.
At epoch 20, baseline online training accuracy is 89.31% but validation accuracy
is 64.94%, below its best epoch 9 score. This indicates overfitting with longer
baseline training. Augmentation gives the higher validation score, consistent with improved
generalization in this comparison. These are single-seed **validation results**, not test-set accuracy.
The API serves the augmentation checkpoint selected from these two experiments.
Per-epoch metrics, training logs, both checkpoints and exact split indices are
included in `models/`. The official test split does not select models or settings.

## Run with Docker

```sh
docker build -t assignment2-fastapi .
docker run --rm -p 8000:80 assignment2-fastapi
```

Open http://127.0.0.1:8000/docs and use POST `/classify` to upload an image.
The container includes trained inference weights and installs the real spaCy
vectors needed by the original Assignment 1 API.

```sh
curl -X POST http://127.0.0.1:8000/classify -F 'image=@examples/test_0_cat.png'
```

The response contains class_name, class_id, confidence and probabilities for
all ten CIFAR-10 classes. Confidence is a softmax probability, not calibrated
certainty. Invalid/missing uploads return 422; oversized uploads return 413.
RGB, grayscale and RGBA images are supported, limited to 5 MB and 20 million pixels.

## Local run and reproduce training

Python 3.13:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Independent fresh runs; each saves best validation weights and complete logs.
python train.py --epochs 20 --output runs/baseline.pt
python train.py --epochs 20 --augmentation --output runs/augmentation.pt
```

Training downloads CIFAR-10 through torchvision. Exact split indices are saved
beside each checkpoint. `.last.pt` includes optimizer and RNG state for optional
continuation with `--resume`; these local continuation files are gitignored.
The API and validation use the shared deterministic transform in app/cnn.py.

## Verification and routes

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover architecture, parameter count, gradients, preprocessing, real-image
inference, upload errors, disjoint stratified splits, checkpoint selection,
resume equivalence, and preserved text generation and real spaCy embeddings.
**23 tests passed** with the supplied model. Docker health is **healthy**.
Docker_Verification.json records actual container HTTP checks and checkpoint
SHA-256 for the supplied augmentation model.

| Method | Route | Purpose |
|---|---|---|
| GET | /health | Service status and classifier readiness |
| GET | /docs | Interactive API documentation |
| GET | /image/classes | Ten labels and input size |
| POST | /classify | Image classification |
| POST | /generate | Assignment 1 bigram generation |
| POST | /embedding | Assignment 1 word embeddings |

## Written answers

`theory/Assignment2_Theory.pdf` contains all six theory answers with calculations.
The Assignment 1 probability solutions are retained under theory/.

Course CNN reference: https://mmerler.github.io/applied_genai_notebooks/Module%204/practical_3_cnn/
The assignment's specified architecture is used throughout.
Prepared with OpenAI Codex assistance.
