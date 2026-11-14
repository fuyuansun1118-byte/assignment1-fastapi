"""Train on all official CIFAR-10 training images; CPU defaults are reproducible."""
import argparse
import torch
from helper_lib.cifar_training import train_cifar

def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['ebm','diffusion'])
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--data-dir',default='data')
    p.add_argument('--output-dir',default='models')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float)
    p.add_argument('--langevin-steps',type=int,default=20)
    p.add_argument('--timesteps',type=int,default=200)
    p.add_argument('--base',type=int,default=16)
    p.add_argument('--resume')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.langevin_steps<1 or a.timesteps<2 or a.base<4 or a.base%4:
        p.error('positive epochs/batch/steps; timesteps>=2; base a multiple of 4')
    torch.set_num_threads(4)
    train_cifar(**vars(a))

if __name__=='__main__':main()
