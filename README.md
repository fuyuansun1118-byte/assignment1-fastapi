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
to tensors with pixel values in [0, 1], without additional normalization. Training and API share `TRANSFORM`.
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

Fresh ten-epoch training with the instructor's tqdm loop, CrossEntropyLoss and
Adam learning rate 0.0005, batch size 32, seed 42. Uses all 50,000 official
training images: 1,563 batches per epoch and 15,630 parameter updates in total.
Final-epoch training accuracy: **76.42%**. The fixed final-epoch checkpoint
achieves **65.34%** on the official separate 10,000-image test set (6,534
correct). Test data is evaluated after training and does not select the epoch.
No validation split or validation-based checkpoint selection is used in this run.
All 21 tests pass; the rebuilt Docker image containing these exact weights also
passes actual HTTP inference, upload validation and health checks.

### Classroom alignment and assignment overrides

The course notebook uses batch size 32, all 50,000 training images, ToTensor,
Adam 0.0005 and ten epochs; those settings and its training loop are followed.
The assignment PDF specifically requires 64 x 64 RGB, two convolutional layers
and a 100-unit hidden fully connected layer. These assignment requirements take
precedence over the classroom notebook's 32 x 32 EnhancedCNN demonstration.
Consequently the only image transform beyond ToTensor is resizing to 64 x 64.
No BatchNorm, Dropout or extra convolution layers are added to the required CNN.

Course source: https://mmerler.github.io/applied_genai_notebooks/Module%204/practical_3_cnn/

The notebook-only `widget.src = losschart(datalogs)` is replaced by saving the
same `datalogs` to `models/cifar10_cnn.training-log.json` (160 records). Training
loss follows the instructor's mean of per-batch losses. Test loss uses a
sample-weighted mean. Checkpoints record their preprocessing configuration, and
the API rejects incompatible checkpoints rather than silently mixing transforms.

## Reproduce training

```sh
python train.py --epochs 10 --device cpu
# On compatible Apple silicon: --device mps
# On a CUDA-enabled system: --device cuda
```

Downloads the official CIFAR-10 Python archive. Defaults: seed 42, batch size 32,
Adam learning rate 0.0005, ten epochs, and the full 50,000-image training split.
The final-epoch checkpoint is evaluated on the separate 10,000-image test split.
The architecture, RGB conversion, resize and tensor conversion are shared with
the API. Metrics and all training logs are saved in `models/`; three real test
images are saved in `examples/`.

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
