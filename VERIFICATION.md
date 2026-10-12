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

## Docker limitation

`docker info` could not connect to a running Docker daemon. Launching an
application named Docker also failed because no application by that name could
be found. Therefore the supplied Dockerfile has NOT been built or tested in
this preparation environment. Install/start Docker Desktop and run the README
commands before claiming a verified container deployment.

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
