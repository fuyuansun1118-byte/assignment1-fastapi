"""Deterministic, request-local latent sampling and PNG grids."""
import io
import math
import torch
from PIL import Image
from torchvision.utils import make_grid

@torch.inference_mode()
def generate_samples(model, device="cpu", num_samples=10, seed=42):
    # CPU-local RNG avoids changing the global RNG during simultaneous requests.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(num_samples, 100, generator=rng).to(device)
    generator = model.generator if hasattr(model, "generator") else model
    was_training = generator.training
    generator.eval()
    try:
        return generator(noise).cpu()
    finally:
        generator.train(was_training)

def samples_to_png(samples):
    pixels = ((samples + 1) / 2).clamp(0, 1)
    grid = make_grid(pixels, nrow=min(8, math.ceil(math.sqrt(len(samples)))), padding=2)
    array = (grid.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
