# Assignment 2: CIFAR-10 CNN with FastAPI

Extends https://github.com/fuyuansun1118-byte/assignment1-fastapi with the exact
CNN required by Assignment2.pdf. Original bigram and word embedding routes are retained.

## Run the supplied trained classifier

Requires Python 3.13. From this project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The included `models/cifar10_cnn.pt` is a trained checkpoint. The server loads it
once on CPU at startup. It also loads the Assignment 1 spaCy vectors. Dependency
installation downloads the spaCy model; normal API inference is local.

Open http://127.0.0.1:8000/docs, select POST /classify, click Try it out,
choose an image, then Execute. Or use:

```sh
curl -X POST http://127.0.0.1:8000/classify \
  -F 'image=@examples/test_0_cat.png'
```

The JSON response contains `class_name`, `class_id`, `confidence` and all ten
class probabilities. The confidence is the softmax probability, not a calibrated
certainty. CIFAR-10 training supports only these ten categories: airplane,
automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Images are decoded, EXIF oriented, converted to RGB, resized to 64 x 64, converted
to tensors and normalized with mean/std 0.5. Training and API share `TRANSFORM`.
Corrupt image uploads return 422, oversized uploads return 413, and missing
checkpoints return 503 for classification. RGB, grayscale and RGBA inputs work.
Uploads are limited to 5 MB and 20 million pixels.

## Architecture

| Layer | Output, channels first (batch excluded) |
|---|---|
| RGB input | 3 x 64 x 64 |
| Conv2d(3,16,3,stride=1,padding=1), ReLU | 16 x 64 x 64 |
| MaxPool2d(2,2) | 16 x 32 x 32 |
| Conv2d(16,32,3,stride=1,padding=1), ReLU | 32 x 32 x 32 |
| MaxPool2d(2,2) | 32 x 16 x 16 |
| Flatten | 8192 |
| Linear(8192,100), ReLU | 100 |
| Linear(100,10) | 10 logits |

825,398 trainable parameters. Training uses CrossEntropyLoss directly on logits;
softmax is applied only when returning inference probabilities. CIFAR-10's native
32 x 32 images are resized to satisfy the assignment's 64 x 64 input requirement.

## Recorded training result

Completed ten epochs. The best validation checkpoint came from epoch 5
(validation accuracy 65.68%). Its accuracy on the separate official 10,000-image
test set is **64.88%** (6,488 correct). All 19 tests passed; a real uvicorn HTTP
classification request also succeeded. Docker was not run because its daemon
was unavailable.

## Reproduce training

```sh
python train.py --epochs 10 --device cpu
# On compatible Apple silicon: --device mps
# On a CUDA-enabled system: --device cuda
```

Downloads the official CIFAR-10 Python archive. Uses seed 42, a fixed 45,000 / 5,000
training/validation split, batch size 128, Adam (learning rate 0.001), and ten
epochs. Selects the checkpoint with best validation accuracy, then evaluates it
on the separate 10,000-image official test set. Test images do not influence
training or checkpoint selection. Recorded measurements are in
`models/cifar10_cnn.metrics.json`. Three held-out images are saved in `examples/`.
No random placeholder weights are served as a trained model.

## Routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Greeting and docs link |
| GET | `/health` | API status and CNN readiness |
| POST | `/generate` | Assignment 1 bigram text generation |
| POST | `/embedding` | Assignment 1 spaCy word vectors |
| GET | `/image/classes` | CIFAR-10 labels and input size |
| POST | `/classify` | Uploaded-image classification |

## Docker

Start Docker Desktop, then:

```sh
docker build -t assignment2-fastapi .
docker run --rm --name assignment2-api -p 8000:80 assignment2-fastapi
```

The image installs dependencies and spaCy during build and copies the already
trained CNN checkpoint. Training and dataset downloads are unnecessary during
container startup. Open http://127.0.0.1:8000/docs, or run the same curl request.
The PyTorch CPU index avoids large CUDA dependencies for an ordinary CPU server.
See VERIFICATION.md for actual execution status of Docker and tests.

## Verification

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests verify the required architecture, parameter count, gradient updates,
classification using the real checkpoint, all ten probabilities, invalid uploads,
and preserved Assignment 1 endpoints using the real spaCy vectors.

## Submission

The rubric requires GitHub code, Docker deployment, callable model inference,
correct architecture and written theory answers. Upload
`theory/Assignment2_Theory.pdf` and provide your accessible GitHub repository URL
according to the course submission form. Include `models/cifar10_cnn.pt` in the
repository so the instructor can run inference without retraining. Ignore local
virtual environments and downloaded CIFAR-10 data.

This project was prepared with OpenAI Codex assistance. Review and understand
its code and written answers before submitting, and follow the course's
assistance-disclosure instructions.
