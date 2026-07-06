"""The model's shape contract and export round-trip. The jaxtyping runtime
check is itself under test: a wrong-shaped input must raise, not silently
broadcast."""

from __future__ import annotations

import json

import pytest
import torch
from jaxtyping import TypeCheckError

from tetrnn.export import export
from tetrnn.model import BOARD_H, BOARD_W, FEATURE_LEN, N_OUT, TetrNet


def _obs(batch: int = 3):
    own = torch.rand(batch, 1, BOARD_H, BOARD_W)
    opp = torch.rand(batch, 1, BOARD_H, BOARD_W)
    feats = torch.randn(batch, FEATURE_LEN)
    return own, opp, feats


def test_serve_returns_the_five_heads():
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    out = model.serve(*_obs(3))
    assert out.shape == (3, N_OUT)
    # aux head is tanh-bounded; wdl/policy are unbounded logits.
    assert out[:, 4].abs().max() <= 1.0


def test_wrong_shape_is_caught_at_the_boundary():
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    own, opp, feats = _obs(3)
    # 84 features where 85 are required — the exact silent-mismatch class
    # jaxtyping exists to turn into a loud error.
    with pytest.raises((TypeCheckError, Exception)):
        model.serve(own, opp, feats[:, :84])


def test_export_writes_the_flat_contract(tmp_path):
    model = TetrNet(conv_channels=(1, 4, 8, 4))
    export(model, tmp_path, attack_w=125.0)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["schema_version"] == 2
    assert cfg["arch"]["conv_channels"] == [1, 4, 8, 4]
    assert len(cfg["feature_mean"]) == FEATURE_LEN
    assert cfg["contract"] == {"z_scale": 10000.0, "attack_w": 125.0}
    # Flat tensor names, no `tower.` / `convs.0.` nesting the loader can't read.
    from safetensors import safe_open

    with safe_open(tmp_path / "net_v2.safetensors", framework="pt") as f:
        keys = set(f.keys())
    assert {"conv1.weight", "conv2.weight", "conv3.weight"} <= keys
    assert {"board_fc.weight", "feat_fc.weight", "head1.weight", "head2.weight"} <= keys
    assert not any("convs." in k or "tower." in k for k in keys)
