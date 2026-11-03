"""Alternating discriminator and generator updates, plus resumable checkpoints."""
import json
import time
from pathlib import Path
import torch
from torch import nn
from .generator import generate_samples, samples_to_png

def train_gan(model, data_loader, criterion=None, optimizer=None, device="cpu",
              epochs=20, output="models/mnist_gan.pt", lr=0.0002, seed=42,
              resume=None):
    model.to(device)
    criterion = criterion or nn.BCELoss()
    if optimizer is None:
        opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    else:
        opt_g, opt_d = optimizer
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history, start = [], 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt_g.load_state_dict(state["optimizer_g"])
        opt_d.load_state_dict(state["optimizer_d"])
        start, history = state["epoch"], state["history"]
        torch.set_rng_state(state["rng_state"])
        data_loader.generator.set_state(state["loader_rng_state"])
        if device == "mps" and "mps_rng_state" in state:
            torch.mps.set_rng_state(state["mps_rng_state"])
    for epoch in range(start, epochs):
        tick = time.monotonic()
        model.train()
        total_d = total_g = count = 0
        for real, _ in data_loader:
            real = real.to(device)
            batch = len(real)
            ones, zeros = torch.ones(batch, 1, device=device), torch.zeros(batch, 1, device=device)
            opt_d.zero_grad(set_to_none=True)
            fake = model.generator(torch.randn(batch, 100, device=device))
            loss_d = (criterion(model.discriminator(real), ones) +
                      criterion(model.discriminator(fake.detach()), zeros)) / 2
            loss_d.backward()
            opt_d.step()
            # Freeze D parameters; retain train-mode BatchNorm for consistent gradients.
            model.discriminator.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            loss_g = criterion(model.discriminator(fake), ones)
            loss_g.backward()
            opt_g.step()
            model.discriminator.requires_grad_(True)
            model.discriminator.train()
            total_d += loss_d.item() * batch
            total_g += loss_g.item() * batch
            count += batch
        row = {"epoch": epoch + 1, "d_loss": total_d/count, "g_loss": total_g/count,
               "images": count, "seconds": round(time.monotonic()-tick, 2)}
        history.append(row)
        model.eval()
        torch.save({"generator": model.generator.cpu().state_dict(), "epoch": epoch+1,
                    "latent_dim": 100, "dataset": "MNIST", "seed": seed,
                    "batch_size": data_loader.batch_size, "learning_rate": lr,
                    "betas": [0.5, 0.999]}, output)
        model.to(device)
        state = {"model": model.state_dict(), "optimizer_g": opt_g.state_dict(),
                 "optimizer_d": opt_d.state_dict(), "epoch": epoch+1,
                 "history": history, "rng_state": torch.get_rng_state(),
                 "loader_rng_state": data_loader.generator.get_state()}
        if device == "mps":
            state["mps_rng_state"] = torch.mps.get_rng_state()
        torch.save(state, output.with_suffix(".last.pt"))
        output.with_suffix(".training-log.json").write_text(json.dumps(history, indent=2))
        preview = output.parent / "gan_samples"
        preview.mkdir(exist_ok=True)
        (preview / f"epoch_{epoch+1:03d}.png").write_bytes(
            samples_to_png(generate_samples(model, device, 64, seed)))
        print(json.dumps(row), flush=True)
    return model
