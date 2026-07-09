"""Read the Rust datagen shards (leapfrog T15).

Each shard is a safetensors file written by `tetr-nn::shards::ShardWriter`
(see crates/tetr-nn/src/shards.rs). Tensors (d decisions, c children total):

  decision     [d, 8]  i32  game_id, seat, ply, played, argmax, z, end_reason, plies_total
  opp_plane    [d, 50] u8   packed 40x10 opponent plane (bit i = cell i)
  parent_own   [d, 50] u8   packed parent own plane (the action head's input)
  parent_feats [d, 85] f32  parent feature vector
  child_offset [d+1]   i32  child_own[child_offset[k]:child_offset[k+1]] are decision k's children
  child_own    [c, 50] u8   packed own plane per child
  child_feats  [c, 85] f32  served feature vector per child
  child_score  [c]     i32  the beam's backed-up root score per child (the completed-Q source)
  child_slot   [c]     u8   action slot (hold*52 + rot*13 + x+2; absent in pre-slot corpora)

The decision's policy target is derived from `child_score` over its children;
the value target is `z`.

GENERATOR-EVAL IDENTITY (the C6 scale-trap guard): `child_score` units depend
on the eval that ran the search — CC2 integer units for CC2-driven seats,
net-contract units (z_scale·ẑ + attack_w·attack) for net seats. The identity
is encoded STRUCTURALLY, not in the schema: the round driver writes per-mode
subdirs (`corpus/cc2/`, `corpus/mirror/`) and the replay-mix symlink names
preserve them (`shard-r8cc2w0-…`, `shard-r8mirrorw0-…`, `shard-base-…`). In
two-arm (`cc2`) games the net seat is `game_id % 2`; mirror games are all-net;
`base` (round-0) corpora are all-CC2. Anything consuming scores in absolute
units (e.g. a value bootstrap) MUST resolve identity through these names —
never mix units through one transform (the round-3 lesson).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from safetensors.numpy import load_file

PACKED_PLANE = 50
FEATURE_LEN = 85
BOARD_H, BOARD_W = 40, 10


@dataclass
class Shard:
    decision: np.ndarray  # [d, 8] i32
    opp_plane: np.ndarray  # [d, 50] u8
    parent_own: np.ndarray | None  # [d, 50] u8 (None on pre-parent corpora)
    parent_feats: np.ndarray | None  # [d, 85] f32
    child_offset: np.ndarray  # [d+1] i32
    child_own: np.ndarray  # [c, 50] u8
    child_feats: np.ndarray  # [c, 85] f32
    child_score: np.ndarray  # [c] i32
    child_slot: np.ndarray | None  # [c] u8, None on pre-slot corpora

    @property
    def n_decisions(self) -> int:
        return self.decision.shape[0]

    def children_of(self, k: int) -> slice:
        return slice(int(self.child_offset[k]), int(self.child_offset[k + 1]))


def read_shard(path: str) -> Shard:
    t = load_file(path)
    return Shard(
        decision=t["decision"].reshape(-1, 8),
        opp_plane=t["opp_plane"].reshape(-1, PACKED_PLANE),
        parent_own=t["parent_own"].reshape(-1, PACKED_PLANE) if "parent_own" in t else None,
        parent_feats=t["parent_feats"].reshape(-1, FEATURE_LEN) if "parent_feats" in t else None,
        child_offset=t["child_offset"].reshape(-1),
        child_own=t["child_own"].reshape(-1, PACKED_PLANE),
        child_feats=t["child_feats"].reshape(-1, FEATURE_LEN),
        child_score=t["child_score"].reshape(-1),
        child_slot=t["child_slot"].reshape(-1) if "child_slot" in t else None,
    )


def shard_paths(dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(dir, "shard-*.safetensors")))


def unpack_plane(packed: np.ndarray) -> np.ndarray:
    """[..., 50] u8 -> [..., 40, 10] f32 occupancy (bit i = row i//10, col i%10)."""
    bits = np.unpackbits(packed, axis=-1, bitorder="little")[..., : BOARD_H * BOARD_W]
    return bits.reshape(*packed.shape[:-1], BOARD_H, BOARD_W).astype(np.float32)


def _summary(dir: str) -> None:
    paths = shard_paths(dir)
    assert paths, f"no shards in {dir}"
    n_dec = 0
    n_child = 0
    z_hist = {-1: 0, 0: 0, 1: 0}
    end_hist: dict[int, int] = {}
    score_min, score_max = np.inf, -np.inf
    death_scores = 0  # roots the beam scored as death-dominated (< -1e6)
    children_per = []
    games = set()
    for p in paths:
        s = read_shard(p)
        n_dec += s.n_decisions
        n_child += s.child_score.shape[0]
        for row in s.decision:
            games.add(int(row[0]))
            z_hist[int(row[5])] = z_hist.get(int(row[5]), 0) + 1
            end_hist[int(row[6])] = end_hist.get(int(row[6]), 0) + 1
        for k in range(s.n_decisions):
            children_per.append(s.children_of(k).stop - s.children_of(k).start)
        live = s.child_score[s.child_score > -1_000_000]
        death_scores += int((s.child_score <= -1_000_000).sum())
        if live.size:
            score_min = min(score_min, int(live.min()))
            score_max = max(score_max, int(live.max()))
    cp = np.array(children_per)
    print(f"dir: {dir}")
    print(f"  shards={len(paths)} games={len(games)} decisions={n_dec} children={n_child}")
    print(f"  children/decision: min={cp.min()} med={int(np.median(cp))} max={cp.max()}")
    print(f"  z histogram (loss/draw/win): {z_hist}")
    print(f"  end_reason histogram (0=topout,1=escalation,2=truecap): {end_hist}")
    print(
        f"  live root score range: [{score_min}, {score_max}]; "
        f"death-dominated children: {death_scores}/{n_child}"
    )
    # Spot-check one decision: obs shapes + that played/argmax index in range.
    s0 = read_shard(paths[0])
    d0 = s0.decision[0]
    ch = s0.children_of(0)
    own = unpack_plane(s0.child_own[ch])
    print(
        f"  decision[0]: game={d0[0]} seat={d0[1]} ply={d0[2]} "
        f"played={d0[3]} argmax={d0[4]} z={d0[5]}"
    )
    print(
        f"    children={ch.stop - ch.start} own_planes={own.shape} "
        f"feats={s0.child_feats[ch].shape}"
    )
    assert d0[3] < ch.stop - ch.start and d0[4] < ch.stop - ch.start, "played/argmax out of range"
    print("  OK: data contract verified")


if __name__ == "__main__":
    import sys

    _summary(sys.argv[1] if len(sys.argv) > 1 else "models/round0_corpus")
