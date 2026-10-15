# Verification record

## Preparation environment

Local Apple silicon macOS, Python 3.13, PyTorch 2.14.1, torchvision 0.29.1,
spaCy 3.8.7 and the real en_core_web_lg 3.8.0 vectors. The environment did not
expose MPS acceleration, so actual CIFAR-10 training uses CPU with four threads.

## Data integrity

Downloaded the original CIFAR-10 Python archive from the MIT-OL-AI-D Hugging Face
mirror, pinned at commit `a48007227f9e2cd0af96175f4afb5ac4965e261b`. Verified the
entire archive against torchvision's official MD5:
`c58f30108f718f92721af3b95e74349a`. torchvision additionally validates individual
batch-file checksums. The mirror was used because the original server was slow.
The ordinary train.py download remains the torchvision official source.

## Checks already executed

- Original Assignment 1 endpoints and real word vectors: 16 tests passed.
- Required CNN layer output shapes and 825,398 parameters: passed.
- A genuine gradient/optimizer update changes convolution weights: passed.
- Actual trained-checkpoint API inference matches direct model inference: passed.
- Valid grayscale/RGBA uploads, invalid uploads, size limit and missing checkpoint: passed.
- Theory PDF: two pages, all six answers present; both pages rendered and visually inspected.

## Docker build and runtime verification

Rebuilt and verified on October 14, 2026 after retraining with the instructor's
loop. Docker Engine 29.8.1, Linux arm64 image.
Image tag: `assignment2-fastapi:teacher-loop`. Container: `assignment2-teacher-validation-20261014`. Host port: 8094.
Docker HEALTHCHECK: **healthy**. Image contents include the exact newly trained
checkpoint, confirmed by comparing its SHA-256 inside and outside the container.

- `/health`: HTTP 200, classifier ready.
- `/docs` and `/openapi.json`: classification route present.
- Actual CIFAR-10 image upload: valid class and ten probabilities summing to one.
- Grayscale/RGBA inputs: HTTP 200; invalid/missing images: 422; oversized file: 413.
- Original `/generate` and real 300-dimensional `/embedding`: working.

Exact image identifier, checkpoint hash and HTTP result: `Docker_Verification.json`.
Other host architectures were not tested in this preparation environment.

## Publication status

Assignment 2 code, trained weights, metrics, examples, tests and theory answers
were published to the existing public GitHub repository using its signed-in web
interface. Git CLI credentials were unavailable. Original Assignment 1 endpoints
and probability solutions are retained. Nothing has been uploaded to Canvas.

## Final training and runtime result

- Fresh model, instructor-provided tqdm loop, Adam learning rate 0.0005.
- Completed all ten CPU training epochs with seed 42 and batch size 128.
- Best validation checkpoint: epoch 10.
- Validation accuracy: 64.16% (5,000 held-out training-split images).
- Official test accuracy: **64.84%** (6,484 / 10,000); loss 1.0390054971694946.
- Final-epoch training accuracy: 76.62%.
- Instructor datalogs: 40 entries (every 100 batches plus each epoch end).
- Full test suite with new checkpoint: **19 passed**.
- Rebuilt Docker image with the new checkpoint: all HTTP/health checks passed.
- Saved actual new-model HTTP response in examples/classification_response.json.
- Per-epoch metrics and training logs are in models/.
