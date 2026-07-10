"""Export a [`TetrNet`][tetrnn.model.TetrNet] to the on-disk format
`crates/tetr-nn/src/net.rs` loads: `net_v2.safetensors` (flat tensor names) +
`config.json` (schema, arch, whitening stats, the leaf contract).

The tensor names are deliberately flat (`conv1`, `board_fc`, `head2`, …) — the
loader performs no permutation, so the export layout *is* the contract. One
function, so the format lives in exactly one place on the Python side.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import TetrNet

Z_SCALE = 10_000.0


def export(model: TetrNet, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()

    tensors = {}
    conv_i = 0
    for key in sd:
        if key.startswith("convs.") and key.endswith(".weight"):
            conv_i += 1
            tensors[f"conv{conv_i}.weight"] = sd[key].contiguous()
            tensors[f"conv{conv_i}.bias"] = sd[key[: -len("weight")] + "bias"].contiguous()
    for name in ("board_fc", "feat_fc", "head1", "head2"):
        tensors[f"{name}.weight"] = sd[f"{name}.weight"].contiguous()
        tensors[f"{name}.bias"] = sd[f"{name}.bias"].contiguous()
    save_file(tensors, str(out / "net_v2.safetensors"))

    config = {
        "schema_version": 3,
        "arch": {"conv_channels": list(model.conv_channels)},
        "feature_mean": model.feat_mean.tolist(),
        "feature_std": model.feat_std.tolist(),
        "heads": {"wdl": [0, 1, 2]},
        "contract": {"z_scale": Z_SCALE},
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))


def load(dir: Path) -> TetrNet:
    """Load an exported model dir back into a [`TetrNet`] (the trainer's
    fine-tune init and the goldens dumper both need the inverse of export)."""
    cfg = json.loads((dir / "config.json").read_text())
    if cfg.get("schema_version") != 3:
        raise ValueError(f"{dir}: config schema_version {cfg.get('schema_version')} != 3")
    model = TetrNet(tuple(cfg["arch"]["conv_channels"]))
    sd = load_file(str(dir / "net_v2.safetensors"))
    renamed = {}
    for k, v in sd.items():
        if k.startswith("conv"):
            n = int(k[4]) - 1
            renamed[f"convs.{2 * n}.{k.split('.', 1)[1]}"] = v
        else:
            renamed[k] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    # The export omits the whitening buffers (they live in config.json).
    assert not unexpected, f"unexpected tensors in {dir}: {unexpected}"
    assert set(missing) <= {"feat_mean", "feat_std"}, f"missing tensors in {dir}: {missing}"
    model.feat_mean.copy_(torch.as_tensor(cfg["feature_mean"], dtype=torch.float32))
    model.feat_std.copy_(torch.as_tensor(cfg["feature_std"], dtype=torch.float32))
    model.eval()
    return model
