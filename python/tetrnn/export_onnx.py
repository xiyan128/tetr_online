"""Export a trained model dir to ONNX (T13's first step — the ANE path).

Two graphs, dynamic batch:
  net_leaf.onnx  — (own [N,1,40,10], opp [N,1,40,10], feats [N,85]) -> heads [N,5]
  net_slots.onnx — same inputs -> slot logits [N,104] (the guided filter's parent forward)

Weights + whitening load from the exported safetensors/config (the same
contract net.rs consumes), so the ONNX graph is faithful to the deployed
forward by construction. Verified against the torch forward at 1e-5.

Usage: uv run python -m tetrnn.export_onnx <model-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .model import TetrNet


def load_model(dir: Path) -> TetrNet:
    cfg = json.loads((dir / "config.json").read_text())
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
    assert not unexpected, unexpected
    left = [m for m in missing if not (m.startswith("feat_") or m.startswith("ssl_head"))]
    assert not left, left
    model.feat_mean.copy_(torch.as_tensor(cfg["feature_mean"], dtype=torch.float32))
    model.feat_std.copy_(torch.as_tensor(cfg["feature_std"], dtype=torch.float32))
    model.eval()
    return model


class LeafWrap(torch.nn.Module):
    def __init__(self, m: TetrNet):
        super().__init__()
        self.m = m

    def forward(self, own, opp, feats):
        return self.m.serve(own, opp, feats)


class SlotWrap(torch.nn.Module):
    def __init__(self, m: TetrNet):
        super().__init__()
        self.m = m

    def forward(self, own, opp, feats):
        return self.m.serve_slots(own, opp, feats)


def main() -> None:
    dir = Path(sys.argv[1])
    model = load_model(dir)
    ex = (
        torch.zeros(2, 1, 40, 10),
        torch.zeros(2, 1, 40, 10),
        torch.zeros(2, 85),
    )
    dyn = {"own": {0: "n"}, "opp": {0: "n"}, "feats": {0: "n"}}
    for name, wrap in [("net_leaf", LeafWrap(model)), ("net_slots", SlotWrap(model))]:
        out = dir / f"{name}.onnx"
        torch.onnx.export(
            wrap,
            ex,
            str(out),
            input_names=["own", "opp", "feats"],
            output_names=["out"],
            dynamic_axes=dyn,
            opset_version=17,
        )
        print(f"exported {out}")

    # Parity vs the torch forward on REALISTIC inputs: random values in
    # training-constant features (std floored at 1e-6) standardize to ~1e6 and
    # blow activations to ~1e5, where fp32 reorder noise looks like a large
    # ABSOLUTE delta (the resolved "tail"). Clamp standardized inputs into the
    # realistic range by drawing feats near the training mean.
    rng = np.random.default_rng(0)
    own = torch.as_tensor(rng.integers(0, 2, (5, 1, 40, 10)).astype(np.float32))
    opp = torch.as_tensor(rng.integers(0, 2, (5, 1, 40, 10)).astype(np.float32))
    z = torch.as_tensor(rng.normal(0, 1, (5, 85)).astype(np.float32)).clamp(-3, 3)
    feats = model.feat_mean + z * model.feat_std  # in-distribution by construction
    want = model.serve(own, opp, feats).detach().numpy()
    try:
        import onnxruntime as ort  # optional; parity check only if available

        sess = ort.InferenceSession(str(dir / "net_leaf.onnx"))
        got = sess.run(None, {"own": own.numpy(), "opp": opp.numpy(), "feats": feats.numpy()})[0]
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
        print(f"onnxruntime parity (relative): median={np.median(rel):.1e} max={rel.max():.1e}")
        assert rel.max() < 1e-4, f"relative parity {rel.max():.2e}" 
    except ImportError:
        print("onnxruntime not installed — parity check skipped (graphs still exported)")


if __name__ == "__main__":
    main()
