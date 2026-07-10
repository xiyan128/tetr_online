"""Regenerate the Python-reference parity fixture the Rust crate tests against.

Builds a small, seeded, non-trivially-whitened net, exports its weights +
config, and dumps its goldens into `crates/tetr-nn/tests/fixtures/pyref/`. The
Rust test `forward_matches_our_python_package` then proves the Rust forward
reproduces this Python forward to 1e-4 — on a model anyone can regenerate from
this script.

Small on purpose (`conv_channels=(1,4,8,4)`): the parity check exercises every
code path at a fraction of the committed weight bytes. Run from `python/`:

    uv run python -m tetrnn.regen_pyref
"""

from __future__ import annotations

from pathlib import Path

import torch

from .export import export
from .goldens import dump
from .model import FEATURE_LEN, TetrNet

FIXTURE = Path(__file__).resolve().parents[2] / "crates/tetr-nn/tests/fixtures/pyref"


def main() -> None:
    torch.manual_seed(20260706)
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    # Non-trivial whitening so the parity check covers that path (not identity).
    gen = torch.Generator().manual_seed(1)
    model.feat_mean.copy_(torch.randn(FEATURE_LEN, generator=gen) * 0.5)
    model.feat_std.copy_(torch.rand(FEATURE_LEN, generator=gen) + 0.5)
    export(model, FIXTURE)
    dump(model, FIXTURE / "golden_v2.json", n=16, seed=7)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
