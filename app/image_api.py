"""Classify an uploaded image using the trained Assignment 2 model."""
import io
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from app.cnn import CLASSES, TRANSFORM

router = APIRouter()
MAX_BYTES = 5 * 1024 * 1024

@router.get('/image/classes')
def image_classes():
    return {'classes': list(CLASSES), 'input_size': [64, 64, 3]}

@router.post('/classify')
def classify(request: Request, image: UploadFile = File(...)):
    model = request.app.state.classifier
    if model is None:
        raise HTTPException(503, 'Trained CNN checkpoint unavailable. Run train.py first.')
    content = image.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, 'Image must be at most 5 MB.')
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 20_000_000:
                raise HTTPException(413, 'Image must contain at most 20 million pixels.')
            source.seek(0)
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            tensor = TRANSFORM(rgb).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(422, 'Upload a valid PNG, JPEG or other supported image.') from error
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    class_id = int(probabilities.argmax())
    return {'class_id': class_id, 'class_name': CLASSES[class_id],
            'confidence': float(probabilities[class_id]),
            'probabilities': {name: float(value) for name, value in zip(CLASSES, probabilities)}}
