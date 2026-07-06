# tetrnn — the Python half of the tetr-nn contract

The net *definition* and *exporter* that produce exactly what
`crates/tetr-nn/src/net.rs` loads (`net_v2.safetensors` + `config.json`). The
Rust crate owns inference and serving; this package owns the model shape and,
later, training. There is one net, defined once, on each side of the language
boundary.

## Tooling

- **uv** — env + lockfile (`uv sync`). `uv.lock` is committed; the env is not.
- **jaxtyping + beartype** — every forward is shape-annotated
  (`Float[Tensor, "batch 85"]`) and the shapes are *runtime-checked*, so a
  wrong-shaped tensor fails at the call site instead of surfacing as a silent
  numeric mismatch downstream.
- **ruff** — lint + format (`uv run ruff check` / `format`).
- **pyright** — types (`uv run pyright`), standard mode, clean.
- **pytest** — `uv run pytest`.

## Cross-language parity

`uv run python -m tetrnn.regen_pyref` regenerates
`crates/tetr-nn/tests/fixtures/pyref/` (a small, seeded net + its goldens). The
Rust test `forward_matches_our_python_package` then proves the Rust forward
reproduces this Python forward to 1e-4 — on a model we can rebuild from source,
unlike the inherited trained `round0` fixture. Change the arch or the forward
on either side and that test tells you the two drifted.
