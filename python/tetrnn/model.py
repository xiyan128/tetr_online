"""The two-board policy+value net, defined once — the Python source of truth
for the weights `crates/tetr-nn/src/net.rs` loads.

The architecture mirrors `net.rs` exactly (a shared conv tower embeds a 40×10
plane; own and opponent planes go through it; a feature branch and the two
tower embeddings meet in a small trunk with five heads). Every forward is
runtime shape-checked with jaxtyping + beartype, so a wrong-shaped tensor
fails loudly at the call site with the expected-vs-actual dims — the class of
bug that used to surface as a silent numeric mismatch three stages later.

The `serve` output is the deployment contract `net.rs` implements: the three
WDL logits and the policy logit are raw; the aux head is `tanh`'d. Nothing
here trains — that is the campaign layer's job; this module only *defines* and
*exports* the net so the Rust loader and a future trainer share one shape.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

# Fixed dims. `conv_channels` is the only arch knob `net.rs` reads from config;
# the embedding/trunk widths are constants on both sides.
BOARD_H, BOARD_W = 40, 10
FEATURE_LEN = 85
BOARD_EMB, FEAT_EMB, TRUNK, N_OUT = 128, 64, 128, 5

# Shape aliases — read as documentation, enforced at runtime.
Plane = Float[Tensor, "batch 1 40 10"]
Feats = Float[Tensor, "batch 85"]
Embedding = Float[Tensor, "batch 128"]
Heads = Float[Tensor, "batch 5"]


class TetrNet(nn.Module):
    """The siamese two-board net. `conv_channels` starts with 1 (single input
    plane) and each pair `(cin, cout)` is a 3×3 pad-1 conv + ReLU."""

    # Buffers, declared so the type checker sees Tensor (nn.Module.__getattr__
    # otherwise widens every attribute to `Tensor | Module`).
    feat_mean: Tensor
    feat_std: Tensor

    def __init__(self, conv_channels: tuple[int, ...] = (1, 16, 32, 32)) -> None:
        super().__init__()
        if len(conv_channels) < 2 or conv_channels[0] != 1:
            raise ValueError(f"conv_channels must start with 1 and have >=1 layer: {conv_channels}")
        self.conv_channels = conv_channels
        convs: list[nn.Module] = []
        for cin, cout in zip(conv_channels, conv_channels[1:], strict=False):
            convs += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
        self.convs = nn.Sequential(*convs)
        self.board_fc = nn.Linear(conv_channels[-1] * BOARD_H * BOARD_W, BOARD_EMB)
        self.feat_fc = nn.Linear(FEATURE_LEN, FEAT_EMB)
        self.head1 = nn.Linear(2 * BOARD_EMB + FEAT_EMB, TRUNK)
        self.head2 = nn.Linear(TRUNK, N_OUT)
        # Feature whitening, applied in the forward and exported to config.json
        # so `net.rs` whitens identically. Trained stats overwrite these.
        self.register_buffer("feat_mean", torch.zeros(FEATURE_LEN))
        self.register_buffer("feat_std", torch.ones(FEATURE_LEN))

    @jaxtyped(typechecker=beartype)
    def embed(self, plane: Plane) -> Embedding:
        """One occupancy plane → its 128-d embedding (the shared tower)."""
        return torch.relu(self.board_fc(self.convs(plane).flatten(1)))

    @jaxtyped(typechecker=beartype)
    def serve(self, own: Plane, opp: Plane, feats: Feats) -> Heads:
        """The deployed forward, matching `net.rs`: wdl+policy raw, aux tanh'd."""
        own_emb = self.embed(own)
        opp_emb = self.embed(opp)
        # Floor the std so a zero-variance (constant) feature can't divide to
        # inf/NaN. net.rs floors identically (MIN_STD), so parity holds.
        f = (feats - self.feat_mean) / self.feat_std.clamp_min(1e-6)
        feat_emb = torch.relu(self.feat_fc(f))
        trunk = torch.relu(self.head1(torch.cat([own_emb, opp_emb, feat_emb], dim=1)))
        raw = self.head2(trunk)
        return torch.cat([raw[:, :4], torch.tanh(raw[:, 4:5])], dim=1)
