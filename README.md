# Assignment 1: FastAPI Word Embeddings and Probability

Extends the Module 3 bigram text-generation API with the spaCy word embedding
functionality demonstrated in Module 2. Theory solutions are in `theory/`.

## Requirements and setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this directory:

```sh
uv sync --locked
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The first sync downloads the course's `en_core_web_lg` 3.8.0 model (about 400 MB)
and other dependencies. Allow sufficient download time and disk space. The
model URL is a project dependency and is recorded in `uv.lock`, so no separate
manual model-install command is needed. Internet access is needed during setup;
normal inference is local. The server loads the model once at startup.

Open http://127.0.0.1:8000/docs to test the API using Swagger UI. Click an endpoint,
choose **Try it out**, enter the JSON request, then click **Execute**.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Greeting and documentation path |
| GET | `/health` | Readiness after the model is loaded |
| POST | `/generate` | Bigram text generation from the classroom sample corpus |
| POST | `/embedding` | Complete word embedding from the course's spaCy model |

### Word embedding

```sh
curl -X POST http://127.0.0.1:8000/embedding \
  -H 'Content-Type: application/json' \
  -d '{"word":"apple"}'
```

The response contains `word`, `model`, `dimensions` (300), and `embedding` (all
300 floating-point coordinates, not a truncated preview). The calculation follows
Practical 3: `nlp(input_word).vector`; `.tolist()` makes the vector JSON serializable.
Only the tokenizer and static vocabulary vectors are needed, so unrelated NLP
pipeline components are excluded when loading the model.

Provide one word of up to 100 characters. Leading/trailing whitespace is removed;
internal whitespace and punctuation are rejected. A value that spaCy splits into
multiple tokens or that has no stored vector returns HTTP 422 with a clear message.
This avoids presenting a zero vector as a meaningful embedding. Case is preserved
for embeddings, consistent with the classroom function.

### Text generation

```sh
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"the","length":20}'
```

Returns `{"generated_text": "..."}`. `length` is the maximum total number of words,
including the starting word, and must be an integer from 1 to 200. Text is lowercased
and tokenized with the classroom regex. Generation samples observed successors
using relative bigram counts. Dividing every count by the same unigram count, as
in the classroom probability formula, gives the same weighted sampling distribution.
Each corpus entry is processed separately to avoid inventing cross-document bigrams.
Generation stops early at a word without successors. An unknown starting word is
returned by itself, following the classroom behavior. Output is stochastic.

## Docker

Install and start [Docker Desktop](https://docs.docker.com/desktop/).
From the project root:

```sh
docker build -t assignment1-fastapi .
docker run --rm --name assignment1-api -p 8000:80 assignment1-fastapi
```

Open the same http://127.0.0.1:8000/docs URL. Stop the local development server first
if port 8000 is occupied. The container listens on port 80; the host uses port 8000.
The build installs the locked dependencies and model. No model download is needed
when starting an already built image. To inspect container readiness:

```sh
docker inspect --format='{{.State.Health.Status}}' assignment1-api
```

The PDFs describe Docker as optional, but the supplied Canvas rubric gives Docker
20 points. A Dockerfile is therefore included. See `VERIFICATION.md` for what has
actually been run in the preparation environment.

## Tests and probability calculations

```sh
uv run pytest -q
uv run python theory/probability_solutions.py
```

Tests use the actual model and check vector contents, validation errors, readiness,
observed bigram transitions, output length, and stopping behavior. Probability
calculations use the standard library, exact fractions where possible, and log base 2
for entropy. `theory/Probability_Solutions.pdf` shows the mathematical steps;
`theory/results.txt` records the script output.

## Files

- `app/main.py`: API schemas, routes, and startup model loading.
- `app/embeddings.py`: spaCy model loading and vector conversion.
- `app/bigram_model.py`: classroom bigram logic adapted into a class.
- `tests/test_api.py`: real-model API and generation tests.
- `pyproject.toml`, `uv.lock`: dependency configuration and reproducible versions.
- `Dockerfile`, `.dockerignore`: container build and runtime.
- `theory/`: written calculations, executable verification, and output.
- `START_HERE_ZH.md`: Chinese setup and submission guide.

## GitHub submission

Create an **empty** repository in your own GitHub account. Replace the placeholders
below with your real username and repository name, then run from this directory:

```sh
git init
git add .
git commit -m "Complete Assignment 1: word embedding API and probability solutions"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Git may ask for your name/email or GitHub authentication; use your own details.
Do not commit your virtual environment, credentials, or downloaded model. They are
excluded or installed separately. Ensure the instructor can access the repository.
Upload the theory PDF to Canvas and include the repository URL with your submission,
following any additional instructor instructions. The provided materials do not
specify exactly where to place the repository URL.

## Sources and assistance

- Supplied `Assignment1-1.pdf`: assignment questions and requirements.
- Supplied Module 3 FastAPI project activity: corpus, `/generate` route and layout.
- Supplied Module 2 Practical 2: tokenizer, bigram counting and weighted sampling.
- Supplied Module 2 Practical 3: `en_core_web_lg` and `nlp(word).vector`.
- [spaCy English model documentation](https://spacy.io/models/en/).
- [uv Docker documentation](https://docs.astral.sh/uv/guides/integration/docker/).

Prepared with AI assistance from OpenAI Codex. Review and understand the code and
calculations, and follow the course's disclosure requirements before submission.
