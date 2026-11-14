"""Trained CIFAR-10 EBM and diffusion image generation endpoints."""
from pathlib import Path
import threading
import torch
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from helper_lib.energy import EnergyModel, langevin_sample
from helper_lib.diffusion import NoiseUNet, diffusion_sample
from helper_lib.generator import samples_to_png

router=APIRouter(tags=['CIFAR-10 generation'])
_generation_slot=threading.BoundedSemaphore(1)

class EnergyRequest(BaseModel):
    num_samples:int=Field(4,ge=1,le=16,strict=True)
    seed:int=Field(42,ge=0,le=2147483647,strict=True)
    steps:int=Field(500,ge=1,le=1000,strict=True)

class DiffusionRequest(BaseModel):
    num_samples:int=Field(4,ge=1,le=16,strict=True)
    seed:int=Field(42,ge=0,le=2147483647,strict=True)
    steps:int=Field(50,ge=2,le=200,strict=True)


def load_cifar_generator(path,kind):
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    if state['config']['kind']!=kind:
        raise ValueError('Checkpoint kind does not match requested generator')
    model=EnergyModel() if kind=='ebm' else NoiseUNet(state['config']['base'])
    model.load_state_dict(state['model'])
    model.eval().requires_grad_(False)
    return {'model':model,'epoch':state['epoch'],'config':state['config'],
            'alpha_bar':state.get('alpha_bar')}


def generate_cifar(bundle,kind,num_samples=4,seed=42,steps=50):
    if kind=='ebm':
        rng=torch.Generator().manual_seed(seed)
        initial=torch.rand((num_samples,3,32,32),generator=rng)*2-1
        return langevin_sample(bundle['model'],initial,steps=steps,rng=rng,
                               step_size=bundle['config']['step_size'],
                               noise_std=bundle['config']['noise_std'])
    return diffusion_sample(bundle['model'],bundle['alpha_bar'],num_samples,steps,seed)


def info(request,kind):
    bundle=getattr(request.app.state,kind)
    return {'kind':kind,'ready':bundle is not None,'dataset':'CIFAR-10',
            'image_shape':[3,32,32], 'trained_epochs':bundle['epoch'] if bundle else 0,
            'conditional':False,'sampler':'Langevin' if kind=='ebm' else 'DDPM ancestral (respaced)'}

@router.get('/ebm/info')
def ebm_info(request:Request):return info(request,'ebm')

@router.get('/diffusion/info')
def diffusion_info(request:Request):return info(request,'diffusion')


def respond(body,request,kind):
    bundle=getattr(request.app.state,kind)
    if bundle is None:
        raise HTTPException(503,f'Trained {kind} weights unavailable. Run train_cifar.py first.')
    if kind=='diffusion' and body.steps>len(bundle['alpha_bar']):
        raise HTTPException(422,'steps must not exceed the trained schedule length')
    if not _generation_slot.acquire(blocking=False):
        raise HTTPException(429,'An image generation request is already running. Retry shortly.')
    try:
        images=generate_cifar(bundle,kind,body.num_samples,body.seed,body.steps)
        return Response(samples_to_png(images),media_type='image/png',
                        headers={'X-Generator':kind,'X-Seed':str(body.seed),'X-Steps':str(body.steps)})
    finally:
        _generation_slot.release()

@router.post('/ebm/generate',response_class=Response,
             responses={200:{'content':{'image/png':{}}}})
def ebm_generate(body:EnergyRequest,request:Request):return respond(body,request,'ebm')

@router.post('/diffusion/generate',response_class=Response,
             responses={200:{'content':{'image/png':{}}}})
def diffusion_generate(body:DiffusionRequest,request:Request):return respond(body,request,'diffusion')
