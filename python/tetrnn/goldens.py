"""Dump cross-language golden vectors: random observations forwarded through a
model, written as `golden_v2.json` in the shape the Rust golden test reads
(`{cases: [{board[400], features[70], out[3]}]}`).

Regenerating a model's goldens and its weights together (see
[`regen_pyref`][tetrnn.regen_pyref]) is what lets the Rust forward be proven
against a Python reference *we control and can reproduce*, instead of an
inherited black-box fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import BOARD_H, BOARD_W, FEATURE_LEN, TetrNet


def dump(model: TetrNet, out: Path, *, n: int = 16, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    cases = []
    with torch.no_grad():
        for _ in range(n):
            board = (torch.rand(1, 1, BOARD_H, BOARD_W, generator=gen) > 0.5).float()
            feats = torch.randn(1, FEATURE_LEN, generator=gen)
            out_vec = model.serve(board, feats)[0]
            cases.append(
                {
                    "board": board.flatten().tolist(),
                    "features": feats.flatten().tolist(),
                    "out": out_vec.tolist(),
                }
            )
    out.write_text(json.dumps({"cases": cases}))
