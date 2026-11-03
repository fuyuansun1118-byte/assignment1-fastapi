import io
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helper_lib.model import Generator, GAN
from helper_lib.generator import generate_samples, samples_to_png
from helper_lib.trainer import train_gan
from app.gan_api import router, load_generator
from torch.utils.data import DataLoader, TensorDataset

def test_shapes_and_gradients():
    torch.set_num_threads(2)
    model = GAN()
    fake = model.generator(torch.randn(4,100))
    assert fake.shape == (4,1,28,28)
    assert fake.min() >= -1 and fake.max() <= 1
    p = model.discriminator(fake)
    assert p.shape == (4,1)
    assert ((p >= 0) & (p <= 1)).all()
    (-p.log().mean()).backward()
    assert all(x.grad is not None and x.grad.isfinite().all() for x in model.parameters())

def test_checkpoint_training_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = GAN()
    before_g = model.generator.layers[0].weight.detach().clone()
    before_d = model.discriminator.layers[0].weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.rand(8,1,28,28)*2-1, torch.zeros(8)),
                        batch_size=4, generator=torch.Generator().manual_seed(7))
    path = tmp_path/'gan.pt'
    train_gan(model, loader, epochs=1, output=path)
    assert not torch.equal(before_g, model.generator.layers[0].weight)
    assert not torch.equal(before_d, model.discriminator.layers[0].weight)
    restored, epoch = load_generator(path)
    assert epoch == 1
    assert torch.equal(generate_samples(model, num_samples=3), generate_samples(restored, num_samples=3))
    train_gan(model, loader, epochs=2, output=path, resume=path.with_suffix('.last.pt'))
    assert load_generator(path)[1] == 2

def test_seed_and_api():
    model = Generator().eval()
    state = torch.get_rng_state().clone()
    one = generate_samples(model, num_samples=4, seed=10)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(one, generate_samples(model, num_samples=4, seed=10))
    assert not torch.equal(one, generate_samples(model, num_samples=4, seed=11))
    app = FastAPI()
    app.include_router(router)
    app.state.gan, app.state.gan_epochs = model, 1
    with TestClient(app) as c:
        result = c.post('/gan/generate', json={'num_samples':4,'seed':10})
        assert result.status_code == 200
        assert result.headers['content-type'] == 'image/png'
        assert result.content == samples_to_png(one)
        assert Image.open(io.BytesIO(result.content)).size == (62,62)
        for body in [{'num_samples':0}, {'num_samples':65}, {'seed':-1}, {'num_samples':1.5}, {'seed':True}]:
            assert c.post('/gan/generate', json=body).status_code == 422
        assert c.get('/gan/info').json()['ready']
        app.state.gan = None
        assert c.post('/gan/generate', json={}).status_code == 503
