"""Full CIFAR-10 EBM and epsilon-prediction diffusion training."""
import copy
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .energy import EnergyModel, langevin_sample
from .diffusion import NoiseUNet, cosine_schedule, forward_noise, diffusion_sample
from .cifar_data import load_cifar_tensor, image_batches
from .generator import samples_to_png


def train_cifar(kind, data_dir="data", output_dir="models", epochs=5,
                batch_size=128, seed=42, lr=None, langevin_steps=20,
                timesteps=200, base=16, resume=None):
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed)
    images=load_cifar_tensor(data_dir,train=True)
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    previews=folder/f"{kind}_samples";previews.mkdir(exist_ok=True)
    history=[];start=0
    config=dict(kind=kind,dataset="CIFAR-10",training_images=len(images),image_shape=[3,32,32],
                batch_size=batch_size,seed=seed,device="cpu",epochs=epochs,
                learning_rate=lr or (0.0001 if kind=="ebm" else 0.0002))
    if kind=="ebm":
        model=EnergyModel()
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=(0.0,0.999))
        replay=torch.rand((4096,3,32,32),generator=rng)*2-1
        config.update(langevin_steps=langevin_steps,step_size=10.0,noise_std=0.005,
                      gradient_clip=0.03,energy_regularization=0.1,replay_size=4096,
                      refresh_probability=0.05)
    elif kind=="diffusion":
        model=NoiseUNet(base)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        beta,alpha_bar=cosine_schedule(timesteps)
        config.update(timesteps=timesteps,base=base,ema_decay=0.995,
                      schedule="cosine",time_embedding="interleaved sin/cos; max_period 10000")
    else:
        raise ValueError("kind must be ebm or diffusion")
    if resume:
        state=torch.load(resume,map_location="cpu",weights_only=True)
        old=state['config']
        for key in ['kind','batch_size','seed','base','timesteps','langevin_steps','learning_rate']:
            if key in old and config.get(key)!=old[key]:
                raise ValueError(f"Resume setting mismatch: {key}")
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.set_state(state['loader_rng'])
        history=state['history'];start=state['epoch']
        if kind=='ebm':replay=state['replay']
        else:ema.load_state_dict(state['ema'])
    for epoch in range(start,epochs):
        tick=time.monotonic();model.train();total=0;metrics={}
        for real in image_batches(images,batch_size,rng):
            n=len(real)
            if kind=="ebm":
                indices=torch.randperm(len(replay),generator=rng)[:n]
                initial=replay[indices].clone()
                refresh=torch.rand(n,generator=rng)<0.05
                initial[refresh]=torch.rand((int(refresh.sum()),3,32,32),generator=rng)*2-1
                model.requires_grad_(False)
                fake=langevin_sample(model,initial,langevin_steps,rng=rng)
                model.requires_grad_(True)
                replay[indices]=fake
                real=(real+torch.randn(real.shape,generator=rng)*0.005).clamp(-1,1)
                e_real=model(real);e_fake=model(fake.detach())
                contrast=e_real.mean()-e_fake.mean()
                reg=0.1*(e_real.square().mean()+e_fake.square().mean())
                loss=contrast+reg
                current={'loss':loss.item(),'contrastive':contrast.item(),
                         'real_energy':e_real.mean().item(),'fake_energy':e_fake.mean().item()}
            else:
                t=torch.randint(timesteps,(n,),generator=rng)
                noise=torch.randn(real.shape,generator=rng)
                noisy=forward_noise(real,t,noise,alpha_bar)
                prediction=model(noisy,t)
                loss=F.mse_loss(prediction,noise)
                current={'noise_mse':loss.item()}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.1 if kind=='ebm' else 1.0)
            optimizer.step()
            if kind=='diffusion':
                with torch.no_grad():
                    for average,param in zip(ema.parameters(),model.parameters()):
                        average.lerp_(param,1-0.995)
            for k,v in current.items():metrics[k]=metrics.get(k,0)+v*n
            total+=n
        row={'epoch':epoch+1,'images':total,**{k:v/total for k,v in metrics.items()},
             'seconds':round(time.monotonic()-tick,2)}
        history.append(row);model.eval()
        inference={'model':(model if kind=='ebm' else ema).state_dict(),
                   'epoch':epoch+1,'config':config}
        if kind=='diffusion':inference['alpha_bar']=alpha_bar
        # Atomic replacement keeps a server from reading a partially written checkpoint.
        path=folder/f'cifar10_{kind}.pt';tmp=path.with_suffix('.tmp')
        torch.save(inference,tmp);tmp.replace(path)
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'epoch':epoch+1,'config':config,'history':history,
               'torch_rng':torch.get_rng_state(),'loader_rng':rng.get_state()}
        if kind=='ebm':state['replay']=replay
        else:state['ema']=ema.state_dict()
        torch.save(state,folder/f'cifar10_{kind}.last.pt')
        (folder/f'cifar10_{kind}.training-log.json').write_text(json.dumps(history,indent=2)+'\n')
        preview_rng=torch.Generator().manual_seed(seed)
        if kind=='ebm':
            samples=langevin_sample(model,torch.rand((16,3,32,32),generator=preview_rng)*2-1,
                                    steps=100,rng=preview_rng)
        else:
            samples=diffusion_sample(ema,alpha_bar,num_samples=16,steps=min(50,timesteps),seed=seed)
        (previews/f'epoch_{epoch+1:03d}.png').write_bytes(samples_to_png(samples))
        print(json.dumps({'kind':kind,**row}),flush=True)
    return model
