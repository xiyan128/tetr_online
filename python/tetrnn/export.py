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

from safetensors.torch import save_file

from .model import TetrNet

Z_SCALE = 10_000.0


def export(model: TetrNet, out: Path, *, attack_w: float = 125.0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()

    tensors = {}
    conv_i = 0
    for key, value in sd.items():
        if key.startswith("convs.") and key.endswith(".weight"):
            conv_i += 1
            tensors[f"conv{conv_i}.weight"] = value.contiguous()
            tensors[f"conv{conv_i}.bias"] = sd[key[: -len("weight")] + "bias"].contiguous()
    for name in ("board_fc", "feat_fc", "head1", "head2", "slot_head"):
        tensors[f"{name}.weight"] = sd[f"{name}.weight"].contiguous()
        tensors[f"{name}.bias"] = sd[f"{name}.bias"].contiguous()
    save_file(tensors, str(out / "net_v2.safetensors"))

    config = {
        "schema_version": 2,
        "arch": {"conv_channels": list(model.conv_channels), "siamese": True},
        "feature_mean": model.feat_mean.tolist(),
        "feature_std": model.feat_std.tolist(),
        "heads": {"wdl": [0, 1, 2], "policy": 3, "aux": 4, "slots": 104},
        "contract": {"z_scale": Z_SCALE, "attack_w": attack_w},
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
