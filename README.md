# Assignment 4: CIFAR-10 EBM and Diffusion Generators

Adds trained Energy-Based and Diffusion image generators to the existing
Assignments 1–3 FastAPI application. The original text, embedding, CNN and GAN
routes are retained. Both new generators operate on native **32×32 RGB CIFAR-10**
images and are unconditional: a request cannot choose a class label.

## Get this version

```sh
git clone --branch assignment4-cifar-generators https://github.com/fuyuansun1118-byte/assignment1-fastapi.git
cd assignment1-fastapi
```

The Assignment 4 implementation lives on `assignment4-cifar-generators`, not
`main`. No cloud inference API or pretrained generative model is used.

## Docker deployment

The inference weights are shipped with the project and copied into the image.
The server neither downloads CIFAR-10 nor trains during startup.

```sh
docker build -t assignment4-fastapi .
docker run --rm -p 8000:80 assignment4-fastapi
```

Open http://127.0.0.1:8000/docs, or query the endpoints:

```sh
curl --fail http://127.0.0.1:8000/ebm/info
curl --fail http://127.0.0.1:8000/diffusion/info
curl --fail -X POST http://127.0.0.1:8000/ebm/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":4,"seed":42,"steps":500}' --output ebm.png
curl --fail -X POST http://127.0.0.1:8000/diffusion/generate \
  -H 'Content-Type: application/json' \
  -d '{"num_samples":4,"seed":42,"steps":50}' --output diffusion.png
python3 verify_cifar_http.py --runtime docker --output Assignment4_Docker_Verification.json
```

The response is an `image/png` grid. Each constituent image is 32×32; two-pixel
padding surrounds the tiles. `num_samples` is an integer 1–16, and `seed` is an
integer 0–2147483647. EBM supports 1–1000 steps (default 500), and diffusion
supports 2–200 steps (default 50). Missing weights return 503, invalid inputs 422,
and an overlapping generation request 429. One new image-generation request
runs at a time per process to bound CPU demand.

`/health` reports `ebm_ready` and `diffusion_ready`. The environment variables
`EBM_MODEL_PATH`, `DIFFUSION_MODEL_PATH`, and `TORCH_NUM_THREADS` can override
paths or CPU threading. Model loading uses `weights_only=True`. Models remain
in eval mode with frozen parameters during serving, while EBM still enables
input gradients for Langevin sampling.

## Reproduce training

Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python train_cifar.py ebm --epochs 3 --data-dir data
python train_cifar.py diffusion --epochs 5 --data-dir data
python -m pytest -q
uvicorn app.main:app --port 8000
```

Both runs use **all 50,000 official training images**, batch size 128, seed 42,
CPU and pixels mapped from [0,255] to [-1,1]. Labels are not used. No data
augmentation, held-out model selection or test-set tuning is performed.
Checkpoints use the final completed epoch, not a selected low-loss preview.

### Energy-Based Model

Adapts the course EBM CNN to three input channels: Conv(3,16,5,2,2),
Conv(16,32,3,2,1), Conv(32,64,3,2,1), Conv(64,64,3,2,1), flatten,
Linear(256,64), Linear(64,1). Hidden activations are SiLU/Swish. Output is a
scalar energy; lower energy is more favorable, with density proportional to
exp(-E(x)). Total parameters: 77,793.

Training minimizes `mean(E(real)) - mean(E(fake))` plus
`0.1 * (mean(E(real)^2) + mean(E(fake)^2))`. Adam uses learning rate 0.0001,
betas (0,0.999), and model-gradient norm clipping at 0.1. A replay buffer of
4096 initially uniform-noise images supplies negative samples, with a 5%
fresh-noise refresh probability. Each training negative receives 20 Langevin
steps: noise std 0.005, step size 10, input-gradient clipping to [-0.03,0.03],
and image clipping to [-1,1]. Real images receive small noise of std 0.005.

The sampler uses `autograd.grad(E(x).sum(), x)` to obtain **input gradients**;
it does not call an optimizer or accumulate model parameter gradients. Fake
images are detached before the parameter-learning step. This sign convention
lowers energy on real images and raises it on negatives. These are finite-step,
clipped Langevin dynamics as in the course practical, not exact equilibrium
samples from a normalized density.

### Diffusion model

A noise-predicting UNet uses residual blocks, GroupNorm, SiLU and projected
sinusoidal time embeddings. Channel widths are 16,32,64,128. Three stride-2
convolutions reduce 32×32 to a 4×4 bottleneck; nearest-neighbor upsampling and
skip concatenation restore 32×32. It has 808,387 parameters. A 64-dimensional,
pair-interleaved sin/cos time embedding uses max period 10000.

Training samples integer timesteps from a 200-step cosine DDPM schedule and
forms `x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon`.
The target is sampled standard-normal epsilon, with mean squared error loss.
Adam uses learning rate 0.0002; model-gradient norm clipping is 1.0. A
parameter EMA (decay 0.995) is saved for inference. GroupNorm has no running
mean/variance buffers to update. The sampler is seeded **ancestral DDPM on a respaced schedule**, normally using 50 evenly spaced indices from the trained 200-step
schedule. Predicted x0 is clipped to [-1,1]. Each reverse transition uses the Gaussian
posterior with effective beta = 1 - alpha_bar_t/alpha_bar_s and adds seeded
posterior noise; the final clean transition adds no noise. A deterministic DDIM
helper is also retained for comparison, but the API and notebook use ancestral
DDPM because its early-checkpoint previews had less residual noise.

This retains the course principles of time-conditioned noise prediction and
reverse diffusion. It uses an explicit discrete cosine schedule and ancestral DDPM
parameterization rather than copying the course continuous-noise-variance
implementation verbatim. The assignment does not prescribe a fixed model
architecture. This design is documented so the time-embedding convention and
sampling equations are unambiguous.

### Checkpoints and continuation

`models/cifar10_ebm.pt` and `models/cifar10_diffusion.pt` hold portable CPU
inference weights, actual epoch and configuration. Metadata files record SHA-256
checksums. Per-epoch logs and fixed-seed previews are stored beside the weights.
The gitignored `.last.pt` files contain model/optimizer/RNG state plus replay
buffer or EMA and are only needed for local continuation:

```sh
python train_cifar.py ebm --epochs 10 --resume models/cifar10_ebm.last.pt
python train_cifar.py diffusion --epochs 16 --resume models/cifar10_diffusion.last.pt
```

`--epochs` is the total target, including completed epochs. Inference checkpoint
writes are atomic. Downloaded data, virtual environments and continuation files
are excluded from the submission ZIP and GitHub.

## Results and limits

Actual completed epoch counts, losses and preview images are supplied in the
training logs and notebook. Contrastive energy losses and noise MSE are training
diagnostics, not image-quality scores or class accuracy. These compact CPU-trained
models can produce textures, colors and ambiguous shapes rather than clearly
recognizable CIFAR-10 objects. No FID, likelihood or semantic-quality threshold
is claimed. A functioning endpoint and successful Docker build do not establish
high perceptual quality.

## Tests and verification

`tests/test_cifar_generation.py` covers input-energy descent without weight or
parameter-gradient changes, sinusoidal values, forward-noise coefficients,
UNet shape/gradients, checkpoint training/resume, seeded PNG generation, request
validation and unavailable checkpoints. The prior assignments' tests remain.

The verification script calls a live HTTP server, checks all model readiness
flags, decodes PNG dimensions, checks same-seed repetition and invalid inputs,
and records actual generation timings. GitHub Actions builds and runs the
Docker image on Linux and uploads its JSON/PNG evidence. Only a successful
workflow execution counts as Docker verification.

## Written work and notebook

`Assignment4.ipynb` contains runnable model/training definitions, recorded full
training results, fresh checkpoint inference and all nine questions. Its HTML
export embeds the results and images for reading. Training is opt-in; normal
Run All loads the supplied weights and performs inference instead of retraining.
`theory/Assignment4_Theory.md` contains the written answers separately.

Prepared with OpenAI Codex assistance. Review the derivations and implementation
before submitting. Include the Assignment 4 branch URL with the files so the
instructor can obtain Docker/API code and weights.

## References

- Course EBM: https://mmerler.github.io/applied_genai_notebooks/Module%208/practical_1_energy_based/
- Course diffusion: https://mmerler.github.io/applied_genai_notebooks/Module%208/practical_2_diffusion/
- Class activity: https://mmerler.github.io/applied_genai_notebooks/Module%208/diffusion_code/
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Cosine schedule / Improved DDPM: https://arxiv.org/abs/2102.09672
- PyTorch tensor: https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html
- PyTorch detach: https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html

## Delivered reference runs

The supplied EBM completed **3 full epochs** (150,000 training-image
presentations), and diffusion completed **5 full epochs** (250,000 presentations).
These are compact CPU reference runs, not pretrained production generators.
Each run uses all 50,000 training images per epoch. The actual losses and
fixed-seed generated outputs are included without claiming high semantic quality.
The supplied notebook displays freshly generated images from these exact
checksum-verified checkpoints. Local and Linux Docker results are stored in
`Assignment4_Verification.json` and `Assignment4_Docker_Verification.json`.

## Final verification and submission files

All **33 local tests passed** with the delivered model files. Real local HTTP
and Linux Docker checks verified all readiness flags, valid PNG generation,
same-seed repetition and HTTP 422 for invalid requests.

- Final Docker run: https://github.com/fuyuansun1118-byte/assignment1-fastapi/actions/runs/43636846039
- Docker-tested source/model commit: `32dc1dae54c3627a3334a98ada49c420e73f3a8c`.
- `Assignment4.ipynb` and `Assignment4.html` include all nine answers, executed
  gradient checks, training history and newly generated checkpoint samples.
- `Assignment4_Docker_Verification.json` and Docker PNGs are downloaded workflow
  evidence for the final 3-/5-epoch checkpoints. Local/Linux PNG hashes can differ
  due to floating-point differences; repeated seeded calls within each runtime
  were verified separately.

The work stays on its own **assignment4-cifar-generators** branch. No pull request
is created for Assignment 4, and nothing is merged into Assignment 3 or main.
