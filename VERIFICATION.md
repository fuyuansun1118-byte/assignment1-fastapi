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

Verified on October 12, 2026 using Docker Desktop on Apple silicon, Docker Engine
29.8.1, Linux arm64 container architecture. No Dockerfile/code changes were needed.

- Built the GitHub project with `docker build -t assignment2-fastapi:verified .`.
- Started container `assignment2-validation-20261012`, mapping localhost:8093 to port 80.
- Docker HEALTHCHECK: **healthy**, container state: **running**.
- `/health`: HTTP 200 and `classifier_ready: true`.
- `/docs` and `/openapi.json`: available; `/classify` is documented.
- `/classify`: actual held-out cat image returned **cat**; ten probabilities sum to one.
- Grayscale and RGBA uploads: HTTP 200.
- Invalid/missing images: HTTP 422; upload over 5 MB: HTTP 413.
- `/generate`: valid text generation response.
- `/embedding`: 300 finite, nonzero coordinates from the real spaCy model.

Exact image identifier and HTTP results are in `Docker_Verification.json`.
The previously unavailable daemon was started before this verification. These
results verify the local Linux arm64 Docker image; other host architectures
have not been run in this preparation environment.

## Publication status

Assignment 2 code, trained weights, metrics, examples, tests and theory answers
were published to the existing public GitHub repository using its signed-in web
interface. Git CLI credentials were unavailable. Original Assignment 1 endpoints
and probability solutions are retained. Nothing has been uploaded to Canvas.

## Final training and runtime result

- Completed all 10 CPU training epochs.
- Best checkpoint: epoch 5.
- Best validation accuracy: 65.68% on 5,000 held-out validation images.
- Official test accuracy: 64.88% (6,488 / 10,000 images); loss 0.9944460801124573.
- Final complete test suite: **19 passed**, with two dependency deprecation warnings.
- Started real uvicorn on localhost:8092 and successfully queried /health and POST /classify via HTTP.
- Saved the sample HTTP response in examples/classification_response.json.
- Full per-epoch measurements are saved in models/cifar10_cnn.metrics.json.
