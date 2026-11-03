"""Train on all 60,000 official MNIST training images; no test-set tuning."""
import argparse
import random
import torch
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_gan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output", default="models/mnist_gan.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("epochs >= 1 and batch-size >= 2 are required")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = get_model("GAN")
    loader = get_data_loader(args.data_dir, args.batch_size, seed=args.seed)
    train_gan(model, loader, device=args.device, epochs=args.epochs,
              output=args.output, lr=args.lr, seed=args.seed, resume=args.resume)

if __name__ == "__main__":
    main()
