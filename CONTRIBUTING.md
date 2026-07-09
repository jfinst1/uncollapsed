# Contributing

Contributions, questions, and pointed critiques are all welcome.

## Dev setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Ground rules
- Keep the core dependency-light (`numpy` only; `matplotlib` is optional).
- Every behavioural claim should be backed by a test. A "maybe" that collapses
  silently is a bug; the same standard applies to the code.
- Run `ruff check .` and `pytest` before opening a PR.

## Good first issues
- Subjective-logic-exact conjunction/disjunction operators in `algebra.py`.
- Unsupervised abstention: hold driven by internal field conflict, not labels.
