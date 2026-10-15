# Verification record

## Architecture and preprocessing

Exact Assignment2.pdf architecture: RGB 64 x 64, Conv(3,16,3,padding=1), ReLU,
MaxPool(2,2), Conv(16,32,3,padding=1), ReLU, MaxPool(2,2), flatten, Linear(8192,100),
ReLU, Linear(100,10). 825,398 parameters. Training and API use the same Resize
and ToTensor pipeline, without additional normalization. Checkpoint preprocessing
metadata is validated at startup to prevent using incompatible older weights.

## Classroom-aligned training

Reconfigured on October 14, 2026 after checking the instructor's actual CNN
notebook. The assignment's architecture and input size override the classroom
EnhancedCNN demo; batch size 32, full training split, ToTensor, Adam 0.0005,
CrossEntropyLoss, ten epochs and the supplied tqdm loop follow that notebook.
The notebook-only live plot is replaced by a JSON log of the same datalogs.

- All 50,000 official training images, batch size 32, seed 42, CPU, four threads.
- 1,563 batches per epoch, ten epochs, 15,630 optimizer updates.
- Fixed final-epoch checkpoint (epoch 10), no validation/test-based selection.
- Final-epoch training accuracy: **76.42%**.
- Official separate test accuracy: **65.34%** (6,534 / 10,000).
- Test cross-entropy loss: 1.05101398229599.
- Per-epoch metrics and 160 datalog entries saved under models/.

Course source: https://mmerler.github.io/applied_genai_notebooks/Module%204/practical_3_cnn/

## Data integrity

Original CIFAR-10 archive downloaded from the MIT-OL-AI-D Hugging Face mirror,
pinned at a48007227f9e2cd0af96175f4afb5ac4965e261b. Archive MD5 matched
c58f30108f718f92721af3b95e74349a; torchvision additionally validates batch files.
train.py normally downloads from torchvision's official source.

## Automated checks

**21 tests passed** with the newly trained checkpoint, including required layer
shapes/parameter count, gradient updates, [0,1] preprocessing, rejection of an
incompatible checkpoint, actual uploaded-image inference, grayscale/RGBA and
invalid uploads, and preserved Assignment 1 routes with real spaCy vectors.

## Docker build and runtime verification

Rebuilt on October 14, 2026. Docker Engine 29.8.1, Linux
arm64. Image `assignment2-fastapi:course-config`, container `assignment2-course-validation-20261014`, host port 8095.
Container healthcheck: **healthy**. SHA-256 inside the image/container matches the
exact checkpoint whose test accuracy is reported above.

Actual HTTP checks: /health reports classifier ready; /docs and OpenAPI expose
/classify; a CIFAR-10 image returns ten valid probabilities and a matching
maximum-probability class; grayscale/RGBA work; corrupt/missing files return 422;
oversized files return 413; /generate works; /embedding returns 300 finite,
nonzero coordinates from the real en_core_web_lg model.
Exact image ID, checkpoint hash and HTTP result: Docker_Verification.json.
Other host architectures have not been tested here.

## Publication and written answers

Updated code, trained weights, logs and evidence are published to the existing
public GitHub repository. Canvas has not been submitted. The original two-page
theory PDF is unchanged: its six answers and architecture match this CNN.
The original Assignment 1 API endpoints and probability solutions are retained.
