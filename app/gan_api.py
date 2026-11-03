"""Generate an MNIST digit grid using a trained GAN, with no runtime download."""
from pathlib import Path
import torch
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from helper_lib.model import Generator
from helper_lib.generator import generate_samples, samples_to_png

router = APIRouter(prefix="/gan", tags=["GAN image generation"])

class GenerationRequest(BaseModel):
    num_samples: int = Field(16, ge=1, le=64, strict=True)
    seed: int = Field(42, ge=0, le=2147483647, strict=True)

def load_generator(path):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = Generator()
    model.load_state_dict(state["generator"])
    model.eval()
    return model, state["epoch"]

@router.get("/info")
def info(request: Request):
    return {"ready": request.app.state.gan is not None, "dataset": "MNIST",
            "latent_dim": 100, "image_shape": [1,28,28],
            "trained_epochs": request.app.state.gan_epochs,
            "conditional": False}

@router.post("/generate", responses={200: {"content": {"image/png": {}}}},
             response_class=Response)
def generate(body: GenerationRequest, request: Request):
    model = request.app.state.gan
    if model is None:
        raise HTTPException(503, "Trained GAN checkpoint missing. Run train_gan.py first.")
    samples = generate_samples(model, "cpu", body.num_samples, body.seed)
    return Response(samples_to_png(samples), media_type="image/png",
                    headers={"X-GAN-Seed": str(body.seed), "X-GAN-Samples": str(body.num_samples)})
