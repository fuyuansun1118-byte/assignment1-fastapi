# Assignment 3: MNIST GAN Image Generation

Extends the existing Assignments 1-2 FastAPI project with the exact Assignment 3
GAN architecture, MNIST training, and an image-generation endpoint. Existing
text generation, word embeddings, and CIFAR-10 classification are retained.

## Required architecture

Generator: noise `(B,100)` -> Linear(100,6272) -> reshape `(B,128,7,7)` ->
ConvTranspose2d(128,64,4,2,1) -> BatchNorm2d(64) -> ReLU ->
ConvTranspose2d(64,1,4,2,1) -> Tanh -> `(B,1,28,28)`.

Discriminator: `(B,1,28,28)` -> Conv2d(1,64,4,2,1) -> LeakyReLU(0.2) ->
Conv2d(64,128,4,2,1) -> BatchNorm2d(128) -> LeakyReLU(0.2) ->
Flatten -> Linear(6272,1) -> Sigmoid (real probability).

Sigmoid pairs with BCELoss. Real and detached generated images train D;
G then minimizes BCE(D(G(z)),1), with D parameters frozen and BatchNorm kept in training mode for that step. This is the non-saturating generator objective. The GAN
is unconditional: it samples digits, and does not accept a requested digit label.

## Run the supplied trained model

Python 3.12; run from this directory. The generator checkpoint is included,
so inference requires neither training nor a runtime MNIST download.

```sh
docker build -t assignment3-fastapi .
docker run --rm -p 8000:80 assignment3-fastapi
```

Open http://127.0.0.1:8000/docs. POST `/gan/generate` returns a PNG grid:

```sh
curl --fail -X POST http://127.0.0.1:8000/gan/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":16,"seed":42}' --output generated_digits.png
curl --fail http://127.0.0.1:8000/gan/info
python3 verify_gan_http.py --runtime docker --output GAN_Docker_Verification.json
```

Each digit is 28x28; the PNG combines digits with two-pixel padding. Sample
count is an integer from 1 to 64; seed is an integer from 0 to 2147483647.
Identical requests reproduce the same PNG in the same runtime. Different
hardware/library versions can produce small numerical differences.
Missing GAN weights produce HTTP 503; malformed requests produce HTTP 422.

## Train or run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python train_gan.py --epochs 5 --batch-size 128 --lr 0.0002 --seed 42
uvicorn app.main:app --port 8000
python -m pytest -q
```

Training uses all 60,000 official MNIST training images, converted to tensors
and normalized from [0,1] to [-1,1]. No augmentation or test-set tuning is used.
Noise is standard normal; both optimizers are Adam with learning rate 0.0002
and betas (0.5,0.999). The default device is CPU; `--device mps` or
`--device cuda` can be selected if available. `--epochs` is the total target
number of epochs, including any resumed epochs.

```sh
python train_gan.py --epochs 20 --resume models/mnist_gan.last.pt
```

`models/mnist_gan.pt` contains portable CPU inference weights and metadata.
The gitignored `.last.pt` contains the complete GAN, optimizers and RNG states
for local continuation; MPS RNG state is included when training on MPS.
Per-epoch losses/times are recorded in `models/mnist_gan.training-log.json`;
fixed-seed grids are saved under `models/gan_samples/`. The final completed
checkpoint is served; GAN losses are not accuracy scores or reliable image
quality metrics. Inspect the grids for quality and diversity. Small MNIST GANs
can still generate distorted or ambiguous digits and can suffer mode collapse.

## Project organization

- `helper_lib/model.py`: Generator, Discriminator, GAN and get_model factory.
- `helper_lib/data_loader.py`: MNIST loading and normalization.
- `helper_lib/trainer.py`: alternating GAN updates, checkpoints and logs.
- `helper_lib/generator.py`: latent sampling and PNG serialization.
- `train_gan.py`: seeded command-line training.
- `app/gan_api.py`: trained checkpoint loading and generation routes.
- `tests/test_gan.py`: gradients, shapes, training, checkpoint/resume, seeds,
  valid PNG output, request validation and missing-weight errors.
- `theory/Assignment3_Theory.pdf`: all seven answers and calculations.
- `.github/workflows/gan-docker.yml`: Linux Docker build and real HTTP smoke test.

## Existing routes

GET `/health`, GET `/image/classes`, POST `/classify`, POST `/generate` and
POST `/embedding` are preserved. GAN uses `/gan/generate` so it does not
conflict with text generation. `/health` reports `gan_ready` as well as
classifier readiness. `GAN_MODEL_PATH` optionally overrides the checkpoint path.
The original spaCy embedding model is installed during Docker build.
See `README_Assignment2.md` for the earlier CNN experiments. The old
`Docker_Verification.json` describes Assignment 2 only, not the new GAN.

## Verification and submission

Local test and HTTP evidence for the final model are recorded separately from
Docker evidence. A Docker workflow run is successful only when its build and
HTTP smoke test actually pass; merely providing a Dockerfile is not proof.

Commit the new code and trained generator to GitHub. For Canvas, include the
theory PDF and a short implementation/results write-up with the GitHub link.
A project ZIP can also be attached if desired; the rubric does not prescribe
an exact archive format. Do not submit downloaded MNIST data, the virtual
environment or local continuation checkpoints.

Course references:
- https://mmerler.github.io/applied_genai_notebooks/Module%206/practical_1_gan/
- https://mmerler.github.io/applied_genai_notebooks/Module%206/gan_code/
- https://mmerler.github.io/applied_genai_notebooks/Module%204/nn_code/

Prepared with OpenAI Codex assistance. Review the code and written derivations
before submitting, and follow your course's permitted-assistance policy.

## Completed Assignment 3 results

The delivered checkpoint completed **5 epochs** (60,000 images per epoch,
300,000 total training-image presentations) on CPU with the settings above.
All **26 tests passed**. `GAN_Verification.json` records real local HTTP checks.
`GAN_Docker_Verification.json` records the successful Linux Docker checks for
the final five-epoch model, including the workflow URL and tested code commit.
Docker build, startup, model readiness, PNG generation and same-seed repetition
all passed. Linux and macOS PNG hashes can differ due to numerical differences;
each environment reproduces its own repeated seeded request.

- Docker run: https://github.com/fuyuansun1118-byte/assignment1-fastapi/actions/runs/41570563421
- Draft PR: https://github.com/fuyuansun1118-byte/assignment1-fastapi/pull/1
- Canvas-ready report: `Assignment3_Submission.pdf` (implementation, results,
  generated examples, and all seven theory questions).

The new code is on the **assignment3-gan** branch, not main. To get this version:

```sh
git clone --branch assignment3-gan https://github.com/fuyuansun1118-byte/assignment1-fastapi.git
cd assignment1-fastapi
```
