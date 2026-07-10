"""The value net, defined once — the Python source of truth for the weights
`crates/tetr-nn/src/net.rs` loads.

The architecture mirrors `net.rs` exactly (parity pinned by the golden fixture
both languages share):

    plane [1,40,10] ─ conv3x3+relu ×3 ─ flatten ─ board_fc+relu ─ 128
    features [70] ─ whiten ─ feat_fc+relu ─ 64
    concat ─ head1+relu ─ head2 ─ [3]  (win/draw/loss logits, raw)

Every forward is runtime shape-checked with jaxtyping + beartype, so a
wrong-shaped tensor fails loudly at the call site — the class of bug that used
to surface as a silent numeric mismatch three stages later.
"""

from __future__ import annotations

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

# Fixed dims. `conv_channels` is the only arch knob `net.rs` reads from config;
# the embedding/trunk widths are constants on both sides.
BOARD_H, BOARD_W = 40, 10
FEATURE_LEN = 70
BOARD_EMB, FEAT_EMB, TRUNK, N_OUT = 128, 64, 128, 3

# Shape aliases — read as documentation, enforced at runtime.
Plane = Float[Tensor, "batch 1 40 10"]
Feats = Float[Tensor, "batch 70"]
Logits = Float[Tensor, "batch 3"]


class TetrNet(nn.Module):
    """`conv_channels` starts with 1 (single input plane) and each pair
    `(cin, cout)` is a 3×3 pad-1 conv + ReLU."""

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
        self.head1 = nn.Linear(BOARD_EMB + FEAT_EMB, TRUNK)
        self.head2 = nn.Linear(TRUNK, N_OUT)
        # Feature whitening, applied in the forward and exported to config.json
        # so `net.rs` whitens identically. Trained stats overwrite these.
        self.register_buffer("feat_mean", torch.zeros(FEATURE_LEN))
        self.register_buffer("feat_std", torch.ones(FEATURE_LEN))

    @jaxtyped(typechecker=beartype)
    def serve(self, board: Plane, feats: Feats) -> Logits:
        """The deployed forward, matching `net.rs`: raw WDL logits."""
        board_emb = torch.relu(self.board_fc(self.convs(board).flatten(1)))
        # Floor the std so a zero-variance (constant) feature can't divide to
        # inf/NaN. net.rs floors identically (MIN_STD), so parity holds.
        f = (feats - self.feat_mean) / self.feat_std.clamp_min(1e-6)
        feat_emb = torch.relu(self.feat_fc(f))
        trunk = torch.relu(self.head1(torch.cat([board_emb, feat_emb], dim=1)))
        return self.head2(trunk)
