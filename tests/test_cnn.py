import io
from pathlib import Path
import torch
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.cnn import AssignmentCNN, CLASSES, TRANSFORM, load_classifier
from app.main import app


def test_required_architecture():
    model = AssignmentCNN()
    shapes = []
    handles = [layer.register_forward_hook(lambda m, inputs, output: shapes.append(tuple(output.shape)))
               for layer in model.features]
    with torch.inference_mode():
        output = model(torch.zeros(2, 3, 64, 64))
    for handle in handles:
        handle.remove()
    assert shapes == [(2,16,64,64), (2,16,64,64), (2,16,32,32),
                      (2,32,32,32), (2,32,32,32), (2,32,16,16)]
    assert output.shape == (2,10)
    assert sum(p.numel() for p in model.parameters()) == 825398


def test_training_step_updates_weights():
    model = AssignmentCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    original = model.features[0].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2,3,64,64)), torch.tensor([1,3]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(original, model.features[0].weight)


def test_course_preprocessing_uses_zero_to_one_pixels():
    black = TRANSFORM(Image.new('RGB', (32, 32), color=(0, 0, 0)))
    white = TRANSFORM(Image.new('RGB', (32, 32), color=(255, 255, 255)))
    assert black.shape == white.shape == (3, 64, 64)
    assert torch.all(black == 0) and torch.all(white == 1)


def test_checkpoint_with_old_preprocessing_is_rejected(tmp_path):
    path = tmp_path / 'old_checkpoint.pt'
    torch.save({'classes': list(CLASSES), 'image_size': 64, 'state_dict': {}}, path)
    with pytest.raises(ValueError, match='preprocessing'):
        load_classifier(path)


def test_real_checkpoint_and_uploaded_image():
    with TestClient(app) as client:
        assert client.get('/image/classes').json()['classes'] == list(CLASSES)
        path = Path('examples/test_0_cat.png')
        response = client.post('/classify', files={'image': ('sample.png',path.read_bytes(),'image/png')})
        assert response.status_code == 200
        body = response.json()
        assert body['class_name'] == CLASSES[body['class_id']]
        assert abs(sum(body['probabilities'].values()) - 1) < 1e-5
        with Image.open(path) as image:
            with torch.inference_mode():
                probs = app.state.classifier(TRANSFORM(image.convert('RGB')).unsqueeze(0)).softmax(1)[0]
        assert body['class_id'] == int(probs.argmax())
        assert abs(body['confidence'] - float(probs.max())) < 1e-6
        for mode in ['L', 'RGBA']:
            stream = io.BytesIO()
            Image.new(mode, (32,32)).save(stream, format='PNG')
            assert client.post('/classify', files={'image': ('image.png',stream.getvalue())}).status_code == 200
        assert client.post('/classify', files={'image': ('bad.png',b'not an image')}).status_code == 422
        assert client.post('/classify', files={'image': ('large.png', b'a' * (5*1024*1024+1))}).status_code == 413
        assert client.post('/classify').status_code == 422
        app.state.classifier = None
        assert client.post('/classify', files={'image': ('image.png',path.read_bytes())}).status_code == 503
