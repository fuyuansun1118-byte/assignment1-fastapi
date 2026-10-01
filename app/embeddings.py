"""Module 2 Practical 3 word vectors, loaded once at server startup."""

import spacy

MODEL_NAME = "en_core_web_lg"


def load_model():
    # Only tokenization and static vectors are needed for this assignment.
    # Excluding these components saves memory without changing word vectors.
    return spacy.load(
        MODEL_NAME,
        exclude=["tok2vec", "tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"],
    )


def calculate_embedding(nlp, input_word: str) -> dict:
    doc = nlp(input_word)
    if len(doc) != 1:
        raise ValueError("Provide one word that spaCy tokenizes as a single token.")
    if not doc.has_vector:
        raise ValueError("This word has no vector in en_core_web_lg. Try a common English word.")
    return {
        "word": input_word,
        "model": MODEL_NAME,
        "dimensions": int(doc.vector.size),
        "embedding": doc.vector.tolist(),
    }
