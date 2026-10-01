"""Integration tests use the real downloaded spaCy model, not mocked vectors."""

import math
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.bigram_model import BigramModel


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_health_and_docs(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/docs").status_code == 200


def test_embedding_matches_classroom_calculation(client):
    response = client.post("/embedding", json={"word": "apple"})
    assert response.status_code == 200
    data = response.json()
    assert data["dimensions"] == 300
    assert len(data["embedding"]) == 300
    assert all(math.isfinite(x) for x in data["embedding"])
    assert any(x != 0 for x in data["embedding"])
    np.testing.assert_allclose(data["embedding"], app.state.nlp("apple").vector)


@pytest.mark.parametrize("word", ["", "   ", "two words", "qzxqzxqzxunknown", "a" * 101])
def test_invalid_word(client, word):
    assert client.post("/embedding", json={"word": word}).status_code == 422


def test_trims_word(client):
    assert client.post("/embedding", json={"word": " apple "}).json()["word"] == "apple"


def test_generation_follows_observed_bigrams(client):
    response = client.post("/generate", json={"start_word": "THE", "length": 20})
    assert response.status_code == 200
    words = response.json()["generated_text"].split()
    assert words[0] == "the"
    assert 1 <= len(words) <= 20
    for a, b in zip(words, words[1:]):
        assert b in app.state.bigram_model.transitions[a]


@pytest.mark.parametrize("length", [0, -1, 201, 1.5, True])
def test_invalid_generation_length(client, length):
    assert client.post("/generate", json={"start_word": "the", "length": length}).status_code == 422


def test_terminal_and_unknown_words(client):
    for word in ["effective", "unknownword"]:
        assert client.post("/generate", json={"start_word": word, "length": 10}).json()["generated_text"] == word


def test_no_cross_document_transitions():
    model = BigramModel(["a b", "c d"])
    assert model.generate_text("a", 20) == "a b"
    assert model.generate_text("c", 1) == "c"
