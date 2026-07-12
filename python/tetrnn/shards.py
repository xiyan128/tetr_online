"""Reader for training shards (schema 3) written by the Rust datagen driver.

One row per played decision:

    decision  [d, 6] i32   game_id, seat, ply, z, end_reason, plies_total
    own       [d, 50] u8   packed 40x10 occupancy plane (LSB-first bits)
    feats     [d, 70] f32  served feature vector
    alt_own   [d, 50] u8   a random NON-best sibling's post-placement plane
    alt_feats [d, 70] f32  that sibling's feature vector
    has_alt   [d]     u8   1 when the decision had >=2 placements

The alt pair carries "the search preferred own over alt_own" — unit-free
within-decision ranking supervision (outcome-only labels measurably cannot
rank sibling placements).

Shard flushes are game-aligned (a game never spans shards). The trainer splits
train/holdout by game_id (not shard position) so the split survives datagen's
work-stealing. The writer stamps a schema tag and an FNV checksum; this reader
rejects anything else loudly.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass

import numpy as np
from safetensors import safe_open

PACKED_PLANE = 50
FEATURE_LEN = 70
BOARD_H, BOARD_W = 40, 10


@dataclass
class Shard:
    decision: np.ndarray  # [d, 6] i32
    own: np.ndarray  # [d, 50] u8
    feats: np.ndarray  # [d, 70] f32
    alt_own: np.ndarray  # [d, 50] u8
    alt_feats: np.ndarray  # [d, 70] f32
    has_alt: np.ndarray  # [d] u8

    @property
    def n_rows(self) -> int:
        return int(self.decision.shape[0])

    @property
    def z(self) -> np.ndarray:
        """Per-row outcome from the mover's perspective (+1/0/-1)."""
        return self.decision[:, 3]

    @property
    def game_id(self) -> np.ndarray:
        return self.decision[:, 0]


def _fnv1a(data: bytes) -> int:
    """FNV-1a 64 — must match `tetr-nn/src/obs.rs::fnv1a` bit for bit."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_shard(path: str, verify: bool = True) -> Shard:
    """Read a shard. `verify=True` runs the FNV-1a payload checksum — a
    pure-Python per-byte loop that is ~99% of this call's cost (85ms vs 0.3ms
    on a 0.9MB shard). The corpus is immutable, so the trainer verifies each
    shard ONCE (epoch 0) and reads `verify=False` thereafter — same corruption
    guarantee, a quarter of the passes."""
    with safe_open(path, framework="np") as f:
        meta = f.metadata() or {}
        if meta.get("schema") != "3":
            raise ValueError(f"{path}: shard schema {meta.get('schema')!r} != '3'")
        decision = f.get_tensor("decision").reshape(-1, 6)
        own = f.get_tensor("own").reshape(-1, PACKED_PLANE)
        feats = f.get_tensor("feats").reshape(-1, FEATURE_LEN)
        alt_own = f.get_tensor("alt_own").reshape(-1, PACKED_PLANE)
        alt_feats = f.get_tensor("alt_feats").reshape(-1, FEATURE_LEN)
        has_alt = f.get_tensor("has_alt").reshape(-1)
        if verify:
            # Payload checksum (XOR of per-tensor FNV-1a over the little-endian
            # bytes) — a corrupt shard fails loudly here, not as garbage labels
            # three stages later.
            computed = (
                _fnv1a(decision.astype("<i4").tobytes())
                ^ _fnv1a(own.tobytes())
                ^ _fnv1a(feats.astype("<f4").tobytes())
                ^ _fnv1a(alt_own.tobytes())
                ^ _fnv1a(alt_feats.astype("<f4").tobytes())
                ^ _fnv1a(has_alt.tobytes())
            )
            stored = meta.get("checksum")
            if stored != f"{computed:016x}":
                raise ValueError(
                    f"{path}: checksum mismatch (stored {stored}, computed {computed:016x})"
                )
        return Shard(
            decision=decision,
            own=own,
            feats=feats,
            alt_own=alt_own,
            alt_feats=alt_feats,
            has_alt=has_alt,
        )


def read_feats(path: str) -> np.ndarray:
    """Just the `feats` tensor `[d, 70]` — for the whitening pass, which needs
    only features. No checksum (epoch 0's full read verifies each shard)."""
    with safe_open(path, framework="np") as f:
        return f.get_tensor("feats").reshape(-1, FEATURE_LEN)


def shard_paths(dir: str) -> list[str]:
    """Every shard under `dir`, recursively (datagen writes under wN/ worker
    subdirs), in sorted order."""
    return sorted(glob.glob(f"{dir}/**/shard-*.safetensors", recursive=True))


def unpack_plane(packed: np.ndarray) -> np.ndarray:
    """[n, 50] u8 → [n, 40, 10] f32 (LSB-first bits, matching the Rust packer)."""
    bits = np.unpackbits(packed, axis=-1, bitorder="little")[..., : BOARD_H * BOARD_W]
    return bits.reshape(*packed.shape[:-1], BOARD_H, BOARD_W).astype(np.float32)


def _summary(dir: str) -> None:
    paths = shard_paths(dir)
    rows = 0
    games: set[int] = set()
    z_counts = {-1: 0, 0: 0, 1: 0}
    for p in paths:
        s = read_shard(p)
        rows += s.n_rows
        games.update(int(g) for g in np.unique(s.game_id))
        for z, n in zip(*np.unique(s.z, return_counts=True), strict=True):
            z_counts[int(z)] += int(n)
    print(f"{dir}: {len(paths)} shards, {rows} rows, {len(games)} games, z {z_counts}")


if __name__ == "__main__":
    _summary(sys.argv[1])
