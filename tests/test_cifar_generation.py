"""Behavioral checks for EBM gradients, diffusion equations, and API failures."""
import io
import math
import torch
import pytest
from PIL import Image
from torch import nn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helper_lib.energy import EnergyModel,langevin_sample
from helper_lib.diffusion import NoiseUNet,SinusoidalTimeEmbedding,cosine_schedule,forward_noise,ddim_sample
from app.cifar_generation_api import router,load_cifar_generator


def test_langevin_descends_input_without_parameter_gradients():
    class Quadratic(nn.Module):
        def __init__(self):
            super().__init__();self.scale=nn.Parameter(torch.tensor(1.0))
        def forward(self,x):return self.scale*x.flatten(1).square().sum(1)
    m=Quadratic();initial=torch.full((2,3,4,4),0.5)
    before=m.scale.detach().clone()
    with torch.no_grad():
        result=langevin_sample(m,initial,steps=5,step_size=0.1,noise_std=0.0)
    assert (m(result)<m(initial)).all()
    assert m.scale.grad is None and torch.equal(m.scale,before)
    assert not result.requires_grad
    assert torch.equal(initial,torch.full_like(initial,0.5))


def test_embedding_and_noise_formula():
    embedding=SinusoidalTimeEmbedding(8)(torch.tensor([1.]))[0]
    expected=torch.tensor([v for f in [1.,.1,.01,.001] for v in [math.sin(f),math.cos(f)]])
    torch.testing.assert_close(embedding,expected)
    with pytest.raises(ValueError):SinusoidalTimeEmbedding(7)
    beta,a=cosine_schedule(200)
    assert ((beta>0)&(beta<1)).all() and (a[1:]<a[:-1]).all()
    assert a[-1]<1e-6
    clean=torch.ones(2,3,32,32);noise=torch.full_like(clean,2.)
    t=torch.tensor([0,100]);y=forward_noise(clean,t,noise,a)
    expected=a[t].sqrt()+2*(1-a[t]).sqrt()
    torch.testing.assert_close(y[:,0,0,0],expected)


def test_unet_shapes_and_gradients():
    torch.set_num_threads(2)
    model=NoiseUNet(base=4)
    bottleneck=[]
    handle=model.mid.register_forward_hook(lambda m,i,o:bottleneck.append(o.shape[-2:]))
    pred=model(torch.randn(2,3,64,64),torch.tensor([0,123]))
    handle.remove()
    assert pred.shape==(2,3,64,64) and bottleneck==[torch.Size([8,8])]
    pred.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


@pytest.mark.parametrize('kind',['ebm','diffusion'])
def test_checkpoint_api_seed_and_validation(tmp_path,kind):
    torch.manual_seed(42)
    model=EnergyModel() if kind=='ebm' else NoiseUNet(base=4)
    state={'model':model.state_dict(),'epoch':1,'config':{'kind':kind,'base':4,'step_size':10.,'noise_std':.005}}
    if kind=='diffusion':state['alpha_bar']=cosine_schedule(10)[1]
    path=tmp_path/f'{kind}.pt';torch.save(state,path)
    bundle=load_cifar_generator(path,kind)
    app=FastAPI();app.include_router(router);setattr(app.state,kind,bundle)
    with TestClient(app) as c:
        body={'num_samples':2,'steps':3,'seed':42}
        r=c.post(f'/{kind}/generate',json=body)
        assert r.status_code==200 and r.headers['content-type']=='image/png'
        assert Image.open(io.BytesIO(r.content)).size==(70,36)
        assert c.post(f'/{kind}/generate',json=body).content==r.content
        assert c.post(f'/{kind}/generate',json=body|{'seed':43}).content!=r.content
        assert c.get(f'/{kind}/info').json()['trained_epochs']==1
        for wrong in [{'num_samples':0},{'num_samples':17},{'seed':True},{'steps':201}]:
            assert c.post(f'/{kind}/generate',json=body|wrong).status_code==422
        if kind=='diffusion':assert c.post('/diffusion/generate',json=body|{'steps':11}).status_code==422
        setattr(app.state,kind,None)
        assert c.post(f'/{kind}/generate',json=body).status_code==503
    assert all(p.grad is None for p in bundle['model'].parameters())


@pytest.mark.parametrize('kind',['ebm','diffusion'])
def test_real_training_updates_and_resume(tmp_path,monkeypatch,kind):
    from helper_lib.cifar_training import train_cifar
    dataset=torch.arange(4*3*32*32,dtype=torch.int32).remainder(256).to(torch.uint8).reshape(4,3,32,32)
    monkeypatch.setattr('helper_lib.cifar_training.load_cifar_tensor',lambda *a,**k:dataset)
    model=train_cifar(kind,output_dir=tmp_path,epochs=1,batch_size=2,langevin_steps=2,timesteps=10,base=4)
    s=torch.load(tmp_path/f'cifar10_{kind}.pt',weights_only=True)
    assert s['epoch']==1 and s['config']['training_images']==4
    weight_key=next(iter(dict(model.named_parameters())))
    first=s['model'][weight_key].clone()
    train_cifar(kind,output_dir=tmp_path,epochs=2,batch_size=2,langevin_steps=2,timesteps=10,base=4,
                resume=tmp_path/f'cifar10_{kind}.last.pt')
    after=torch.load(tmp_path/f'cifar10_{kind}.pt',weights_only=True)
    assert after['epoch']==2
    assert not torch.equal(first,after['model'][weight_key])
