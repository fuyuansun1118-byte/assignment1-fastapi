# Verification

Verified locally on 2026-10-01 using macOS arm64 and Python 3.12.14.

## Passed

- Dependencies resolved and installed; `uv.lock` includes the classroom model.
- `pytest`: **16 passed** using the actual `en_core_web_lg` 3.8.0 model.
- A real Uvicorn server responded over HTTP with status 200 for `/health`, `/docs`,
  `/openapi.json`, `POST /embedding`, and `POST /generate`.
- `apple` returned exactly 300 finite embedding coordinates. Tests compared the
  response with the classroom calculation `nlp("apple").vector`.
- All six probability answers were reproduced by `theory/probability_solutions.py`.
  Assertions passed; the variance was also checked with its direct definition.
- The three-page theory PDF was rendered and visually inspected.

- Docker image build succeeded (user-provided build log). The running
  `assignment1-api` container was independently checked with `docker exec`: `/health`,
  `POST /embedding`, and `POST /generate` all returned HTTP 200 on 2026-10-01.

## Observed warning

The installed Starlette test client emits a deprecation warning about its httpx
adapter. All tests pass; this warning concerns the test harness, not inference.

## Not verified / still required

- **GitHub repository:** https://github.com/fuyuansun1118-byte/assignment1-fastapi
- **Canvas:** no submission has been made.

## Numerical answers

| Question | Answer |
|---|---|
| 1(a), 1(b) | 0.12; 0.58 |
| 2 | Not independent |
| 3 | 15/19 = 0.789473684... |
| 4 | 19/117 = 0.162393162... (16.2393%) |
| 5(a), 5(b), 5(c) | 90; 25; 90 |
| 6(a), 6(b) | 1.84643934 bits; 2 bits |
