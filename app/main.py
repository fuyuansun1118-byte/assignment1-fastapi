"""Run from the project root: uv run uvicorn app.main:app --port 8000."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from app.bigram_model import BigramModel
from app.embeddings import calculate_embedding, load_model

# Sample corpus supplied in the Module 3 class activity.
CORPUS = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nlp = load_model()
    app.state.bigram_model = BigramModel(CORPUS)
    yield
    del app.state.nlp


app = FastAPI(
    title="Assignment 1: Text Generation and Word Embeddings",
    version="1.0.0",
    description="Module 3 text generation extended with Module 2 spaCy word embeddings.",
    lifespan=lifespan,
)

SingleWord = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100, pattern=r"^\w+$")]


class TextGenerationRequest(BaseModel):
    start_word: SingleWord = Field(examples=["the"])
    length: int = Field(default=20, ge=1, le=200, strict=True)


class EmbeddingRequest(BaseModel):
    word: SingleWord = Field(examples=["apple"])


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    dimensions: int
    embedding: list[float]


@app.get("/")
def read_root():
    return {"Hello": "World", "docs": "/docs"}


@app.get("/health")
def health():
    # Requests are served only after the lifespan model load succeeds.
    return {"status": "ok", "model": "en_core_web_lg"}


@app.post("/generate")
def generate_text(body: TextGenerationRequest, request: Request):
    text = request.app.state.bigram_model.generate_text(body.start_word, body.length)
    return {"generated_text": text}


@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(body: EmbeddingRequest, request: Request):
    """Return the complete static vector; reject unknown or multi-token words."""
    try:
        return calculate_embedding(request.app.state.nlp, body.word)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
