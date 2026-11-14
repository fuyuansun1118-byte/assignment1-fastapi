"""Noise-predicting UNet, cosine DDPM schedule, and deterministic DDIM sampling."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64, max_period=10000):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("Embedding dimension must be positive and even.")
        self.dim=dim
        self.register_buffer("frequencies", torch.exp(
            -math.log(max_period)*torch.arange(dim//2,dtype=torch.float32)/(dim//2)))
    def forward(self,t):
        angles=t.float().reshape(-1,1)*self.frequencies.reshape(1,-1)
        # Pair-interleaved sin/cos matches the stated convention in theory answers.
        return torch.stack((angles.sin(), angles.cos()),dim=-1).flatten(1)

class TimeBlock(nn.Module):
    def __init__(self,inc,outc,tdim=64):
        super().__init__()
        self.norm1=nn.GroupNorm(4,inc)
        self.conv1=nn.Conv2d(inc,outc,3,padding=1)
        self.time=nn.Linear(tdim,outc)
        self.norm2=nn.GroupNorm(4,outc)
        self.conv2=nn.Conv2d(outc,outc,3,padding=1)
        self.skip=nn.Conv2d(inc,outc,1) if inc!=outc else nn.Identity()
    def forward(self,x,t):
        h=self.conv1(F.silu(self.norm1(x)))+self.time(t)[:,:,None,None]
        return (self.conv2(F.silu(self.norm2(h)))+self.skip(x))/math.sqrt(2)

class NoiseUNet(nn.Module):
    def __init__(self,base=16):
        super().__init__()
        self.base=base
        self.time=nn.Sequential(SinusoidalTimeEmbedding(64),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,64))
        self.input=nn.Conv2d(3,base,3,padding=1)
        self.b0=TimeBlock(base,base)
        self.d1=nn.Conv2d(base,base*2,4,2,1); self.b1=TimeBlock(base*2,base*2)
        self.d2=nn.Conv2d(base*2,base*4,4,2,1); self.b2=TimeBlock(base*4,base*4)
        self.d3=nn.Conv2d(base*4,base*8,4,2,1); self.mid=TimeBlock(base*8,base*8)
        self.u2=TimeBlock(base*8+base*4,base*4)
        self.u1=TimeBlock(base*4+base*2,base*2)
        self.u0=TimeBlock(base*2+base,base)
        self.out=nn.Sequential(nn.GroupNorm(4,base),nn.SiLU(),nn.Conv2d(base,3,3,padding=1))
    def forward(self,x,t):
        e=self.time(t)
        s0=self.b0(self.input(x),e)
        s1=self.b1(self.d1(s0),e)
        s2=self.b2(self.d2(s1),e)
        h=self.mid(self.d3(s2),e)
        h=self.u2(torch.cat((F.interpolate(h,size=s2.shape[-2:],mode="nearest"),s2),1),e)
        h=self.u1(torch.cat((F.interpolate(h,size=s1.shape[-2:],mode="nearest"),s1),1),e)
        h=self.u0(torch.cat((F.interpolate(h,size=s0.shape[-2:],mode="nearest"),s0),1),e)
        return self.out(h)

def cosine_schedule(timesteps=200):
    # Improved-DDPM cosine cumulative alpha; final beta clipped below 1.
    t=torch.linspace(0,timesteps,timesteps+1,dtype=torch.float64)/timesteps
    a=torch.cos((t+0.008)/1.008*math.pi/2).square()
    a=a/a[0]
    beta=(1-a[1:]/a[:-1]).clamp(0.0001,0.999).float()
    return beta, (1-beta).cumprod(0)

def forward_noise(clean,t,noise,alpha_bar):
    a=alpha_bar[t].reshape(-1,1,1,1)
    return a.sqrt()*clean+(1-a).sqrt()*noise

@torch.inference_mode()
def ddim_sample(model,alpha_bar,num_samples=16,steps=50,seed=42):
    if not 2<=steps<=len(alpha_bar):
        raise ValueError("steps must be between 2 and schedule length")
    device=next(model.parameters()).device
    rng=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((num_samples,3,32,32),generator=rng,device=device)
    indices=torch.linspace(len(alpha_bar)-1,0,steps).round().long().tolist()
    for k,index in enumerate(indices):
        a=alpha_bar[index]
        t=torch.full((num_samples,),index,device=device,dtype=torch.long)
        noise=model(x,t)
        clean=((x-(1-a).sqrt()*noise)/a.sqrt()).clamp(-1,1)
        if k==len(indices)-1:
            x=clean
        else:
            next_a=alpha_bar[indices[k+1]]
            # Recompute noise after clipping x0 (DDIM, eta=0).
            noise=(x-a.sqrt()*clean)/(1-a).sqrt()
            x=next_a.sqrt()*clean+(1-next_a).sqrt()*noise
    return x.clamp(-1,1)


@torch.inference_mode()
def diffusion_sample(model,alpha_bar,num_samples=16,steps=50,seed=42):
    """Ancestral DDPM posterior sampling on a respaced cumulative-alpha schedule.

    For a jump from t to s<t, use effective beta=1-alpha_bar[t]/alpha_bar[s].
    The final jump to clean space uses alpha_bar[s]=1 and zero posterior noise.
    """
    if not 2<=steps<=len(alpha_bar):
        raise ValueError("steps must be between 2 and schedule length")
    device=next(model.parameters()).device
    rng=torch.Generator(device=device).manual_seed(seed)
    x=torch.randn((num_samples,3,32,32),generator=rng,device=device)
    indices=torch.linspace(len(alpha_bar)-1,0,steps).round().long().tolist()
    for k,index in enumerate(indices):
        at=alpha_bar[index]
        ts=torch.full((num_samples,),index,device=device,dtype=torch.long)
        eps=model(x,ts)
        clean=((x-(1-at).sqrt()*eps)/at.sqrt()).clamp(-1,1)
        if k==len(indices)-1:
            x=clean
        else:
            a_previous=alpha_bar[indices[k+1]]
            effective_beta=1-at/a_previous
            c_clean=a_previous.sqrt()*effective_beta/(1-at)
            c_noisy=(at/a_previous).sqrt()*(1-a_previous)/(1-at)
            variance=effective_beta*(1-a_previous)/(1-at)
            noise=torch.randn(x.shape,generator=rng,device=device)
            x=c_clean*clean+c_noisy*x+variance.clamp_min(0).sqrt()*noise
    return x.clamp(-1,1)
