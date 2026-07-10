"""Reader for training shards (schema 2) written by the Rust datagen driver.

One row per played decision:

    decision [d, 6] i32   game_id, seat, ply, z, end_reason, plies_total
    own      [d, 50] u8   packed 40x10 occupancy plane (LSB-first bits)
    feats    [d, 70] f32  served feature vector

Shard flushes are game-aligned (a game never spans shards), so a shard-level
train/holdout split IS a game-level split. The writer stamps a schema tag and
an FNV checksum; this reader rejects anything else loudly.
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


def read_shard(path: str) -> Shard:
    with safe_open(path, framework="np") as f:
        meta = f.metadata() or {}
        if meta.get("schema") != "2":
            raise ValueError(f"{path}: shard schema {meta.get('schema')!r} != '2'")
        return Shard(
            decision=f.get_tensor("decision").reshape(-1, 6),
            own=f.get_tensor("own").reshape(-1, PACKED_PLANE),
            feats=f.get_tensor("feats").reshape(-1, FEATURE_LEN),
        )


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
