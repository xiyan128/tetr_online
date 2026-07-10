# tetrnn — the Python half of the learning loop

Four small modules, mirroring the plan in
[`wayfinder/leapfrog/map.md`](../wayfinder/leapfrog/map.md):

- **`model.py`** — the value net (board + 70 features → win/draw/loss),
  defined once as the source of truth for the weights
  `crates/tetr-nn/src/net.rs` loads.
- **`train.py`** — the trainer: WDL cross-entropy on played states, holdout by
  game, per-epoch export. `uv run python -m tetrnn.train <corpus…> <out>`.
- **`round.py`** — one command per expert-iteration round:
  datagen → train (+ replay) → duel + anchor gate. See `--help`.
- **`shards.py`** — the reader for the Rust datagen's shards
  (schema- and checksum-verified).

`export.py` writes (and `load`s back) the on-disk model contract; `goldens.py`
+ `regen_pyref.py` maintain the cross-language parity fixture.

## Tooling

- **uv** — env + lockfile (`uv sync`). `uv.lock` is committed; the env is not.
- **jaxtyping + beartype** — every forward is shape-annotated
  (`Float[Tensor, "batch 70"]`) and the shapes are *runtime-checked*, so a
  wrong-shaped tensor fails at the call site instead of surfacing as a silent
  numeric mismatch downstream.
- **ruff** — lint + format (`uv run ruff check` / `format`).
- **pyright** — types (`uv run pyright`), standard mode, clean.
- **pytest** — `uv run pytest`.

## Cross-language parity

`uv run python -m tetrnn.regen_pyref` regenerates
`crates/tetr-nn/tests/fixtures/pyref/` (a small, seeded net + its goldens). The
Rust test `forward_matches_our_python_package` then proves the Rust forward
reproduces this Python forward to 1e-4 — on a model anyone can rebuild from
source. Change the arch or the forward on either side and that test tells you
the two drifted.
