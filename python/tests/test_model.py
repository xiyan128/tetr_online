"""The model's shape contract and export round-trip. The jaxtyping runtime
check is itself under test: a wrong-shaped input must raise, not silently
broadcast."""

from __future__ import annotations

import json

import pytest
import torch
from jaxtyping import TypeCheckError

from tetrnn.export import export, load
from tetrnn.model import BOARD_H, BOARD_W, FEATURE_LEN, N_OUT, TetrNet


def _obs(batch: int = 3):
    board = torch.rand(batch, 1, BOARD_H, BOARD_W)
    feats = torch.randn(batch, FEATURE_LEN)
    return board, feats


def test_serve_returns_wdl_logits():
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    out = model.serve(*_obs(3))
    assert out.shape == (3, N_OUT)


def test_wrong_shape_is_caught_at_the_boundary():
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    board, feats = _obs(3)
    # 69 features where 70 are required — the exact silent-mismatch class
    # jaxtyping exists to turn into a loud error.
    with pytest.raises(TypeCheckError):
        model.serve(board, feats[:, :-1])


def test_export_writes_the_flat_contract(tmp_path):
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    export(model, tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["schema_version"] == 3
    assert cfg["arch"]["conv_channels"] == [1, 4, 8, 4]
    assert len(cfg["feature_mean"]) == FEATURE_LEN
    assert cfg["contract"] == {"z_scale": 10000.0}
    # Flat tensor names, no `tower.` / `convs.0.` nesting the loader can't read.
    from safetensors import safe_open

    with safe_open(tmp_path / "net_v2.safetensors", framework="pt") as f:
        keys = set(f.keys())
    assert {"conv1.weight", "conv2.weight", "conv3.weight"} <= keys
    assert {"board_fc.weight", "feat_fc.weight", "head1.weight", "head2.weight"} <= keys
    assert not any("convs." in k or "tower." in k for k in keys)


def test_export_load_round_trips_the_forward(tmp_path):
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    model.feat_mean.copy_(torch.randn(FEATURE_LEN))
    model.feat_std.copy_(torch.rand(FEATURE_LEN) + 0.5)
    model.eval()
    export(model, tmp_path)
    back = load(tmp_path)
    board, feats = _obs(4)
    with torch.no_grad():
        want = model.serve(board, feats)
        got = back.serve(board, feats)
    assert torch.allclose(want, got, atol=1e-6)
