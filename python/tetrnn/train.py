"""Round-0 trainer (leapfrog T15): BC on datagen shards with completed-Q
policy targets + terminal-WDL value.

Losses per decision (a sibling group of children):
  * policy: CE(pi', softmax over the group of per-child policy logits), where
    pi' = completed_q_target(child_score) — see targets.py. Dead-coded roots
    carry pi'=0 and contribute no gradient mass.
  * value: CE on the PLAYED child's WDL logits vs z (win=0 / draw=1 / loss=2)
    from the mover's perspective — the state actually visited gets the label;
    counterfactual siblings do not.

The corpus streams SHARD BY SHARD (a full 2000-game corpus is ~13 GB unpacked
— never resident). Shard flushes are game-aligned, so a game never spans
shards and a shard-level split IS a game-level split (the postmortem lesson —
decision-level splits leak). Holdout = every 10th shard.

Usage:
  uv run python -m tetrnn.train <corpus-dir> <out-model-dir> [epochs]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from .export import export
from .model import N_SLOTS, TetrNet
from .shards import Shard, read_shard, shard_paths, unpack_plane
from .targets import completed_q_target, n_eff

BATCH_DECISIONS = 64
LR = 1e-3
SEED = 0
HOLDOUT_EVERY = 10  # every 10th shard is holdout (game-aligned => game-level)


def z_class(z: int) -> int:
    """z=+1 -> win(0), z=0 -> draw(1), z=-1 -> loss(2) (net.rs head order)."""
    return {1: 0, 0: 1, -1: 2}[int(z)]


def shard_batches(shard: Shard, rng: np.random.Generator | None):
    order = np.arange(shard.n_decisions)
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), BATCH_DECISIONS):
        yield order[i : i + BATCH_DECISIONS]


def run_batch(
    model: TetrNet,
    shard: Shard,
    targets: list[np.ndarray],
    dec_idx: np.ndarray,
    device: torch.device,
    metrics: bool,
    live_logits: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Forward one batch of decisions from one shard; return (loss, metrics)."""
    groups, rows, opp_rows, played_local, zc = [], [], [], [], []
    for g, d in enumerate(dec_idx):
        lo, hi = int(shard.child_offset[d]), int(shard.child_offset[d + 1])
        rows.extend(range(lo, hi))
        groups.extend([g] * (hi - lo))
        opp_rows.extend([d] * (hi - lo))
        played_local.append(len(rows) - (hi - lo) + int(shard.decision[d, 3]))
        zc.append(z_class(shard.decision[d, 5]))
    rows = np.asarray(rows)
    groups_np = np.asarray(groups)
    groups_t = torch.as_tensor(groups_np, device=device)

    own = torch.as_tensor(unpack_plane(shard.child_own[rows]), device=device).unsqueeze(1)
    opp = torch.as_tensor(unpack_plane(shard.opp_plane[opp_rows]), device=device).unsqueeze(1)
    feats = torch.as_tensor(shard.child_feats[rows], device=device)
    heads = model.serve(own, opp, feats)

    # Policy: grouped log-softmax over each decision's children.
    logit = heads[:, 3]
    n_groups = len(dec_idx)
    gmax = torch.full((n_groups,), -torch.inf, device=device)
    gmax.scatter_reduce_(0, groups_t, logit, reduce="amax")
    ex = torch.exp(logit - gmax[groups_t])
    gsum = torch.zeros(n_groups, device=device).scatter_add_(0, groups_t, ex)
    logp = logit - gmax[groups_t] - torch.log(gsum[groups_t])
    if live_logits:
        # Round-1 reanalyze form: pi' = softmax(CURRENT logits + c*qnorm),
        # logits detached (the target must not chase its own gradient).
        lg_np = logit.detach().cpu().numpy()
        live_targets: list[np.ndarray] = []
        pos = 0
        for d in dec_idx:
            n_ch = int(shard.child_offset[d + 1]) - int(shard.child_offset[d])
            sc = shard.child_score[shard.children_of(int(d))]
            live_targets.append(
                completed_q_target(sc, logits=lg_np[pos : pos + n_ch])
            )
            pos += n_ch
        batch_targets = {int(d): t for d, t in zip(dec_idx, live_targets)}
    else:
        batch_targets = {int(d): targets[d] for d in dec_idx}
    pi = torch.as_tensor(
        np.concatenate([batch_targets[int(d)] for d in dec_idx]).astype(np.float32),
        device=device,
    )
    policy_ce = -(pi * logp).sum() / n_groups

    # Value: WDL CE on the played child of each decision.
    played_rows = torch.as_tensor(np.asarray(played_local), device=device)
    wdl = heads[played_rows][:, :3]
    z_t = torch.as_tensor(np.asarray(zc), device=device)
    value_ce = torch.nn.functional.cross_entropy(wdl, z_t)

    # Action head: CE(slot-scattered pi', log_softmax(slot logits(parent))).
    # Collided slots (rare same-(rot,x) placements) sum their target mass.
    slot_ce = torch.zeros((), device=device)
    if shard.parent_own is not None and shard.child_slot is not None:
        p_own = torch.as_tensor(unpack_plane(shard.parent_own[dec_idx]), device=device).unsqueeze(1)
        p_opp = torch.as_tensor(unpack_plane(shard.opp_plane[dec_idx]), device=device).unsqueeze(1)
        p_feats = torch.as_tensor(shard.parent_feats[dec_idx], device=device)
        slot_logits = model.serve_slots(p_own, p_opp, p_feats)
        slot_target = np.zeros((n_groups, N_SLOTS), dtype=np.float32)
        for g, d in enumerate(dec_idx):
            t = batch_targets[int(d)]
            lo = int(shard.child_offset[d])
            slots = shard.child_slot[lo : lo + len(t)]
            np.add.at(slot_target[g], slots, t.astype(np.float32))
        st = torch.as_tensor(slot_target, device=device)
        slot_ce = -(st * torch.log_softmax(slot_logits, dim=1)).sum(dim=1).mean()

    m = {
        "policy_ce": float(policy_ce.detach()),
        "value_ce": float(value_ce.detach()),
        "slot_ce": float(slot_ce.detach()),
    }
    if metrics:
        # top-1 agreement with the search argmax; z_hat spread (start-gate).
        with torch.no_grad():
            lg = logit.detach().cpu().numpy()
            arg_local = np.zeros(n_groups, dtype=np.int64)
            for g in range(n_groups):
                sel = np.flatnonzero(groups_np == g)
                arg_local[g] = int(sel[lg[sel].argmax()] - sel[0])
            search_arg = shard.decision[dec_idx, 4]
            m["top1"] = float((arg_local == search_arg).mean())
            p = torch.softmax(wdl, dim=1)
            z_hat = (p[:, 0] - p[:, 2]).detach().cpu().numpy()
            m["z_hat_std"] = float(np.std(z_hat))

    return policy_ce + value_ce + slot_ce, m


def shard_targets(shard: Shard) -> list[np.ndarray]:
    return [
        completed_q_target(shard.child_score[shard.children_of(d)])
        for d in range(shard.n_decisions)
    ]


def whitening_stats(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over all training children, one streaming pass (Welford-ish
    via sum/sumsq — features are bounded so this is numerically fine)."""
    n, s, s2 = 0, 0.0, 0.0
    for p in paths:
        f = read_shard(p).child_feats.astype(np.float64)
        n += f.shape[0]
        s = s + f.sum(axis=0)
        s2 = s2 + (f * f).sum(axis=0)
    mean = s / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def main() -> None:
    corpus_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    live = len(sys.argv) > 4 and sys.argv[4] == "live"
    if live:
        # ROUND-1 POSTMORTEM (2026-07-08): this mode is UNSOUND as implemented —
        # it re-mixes the stored (old) search Q with the TRAINEE's ever-changing
        # logits each batch, a self-amplifying runaway (policy collapsed 0-64 vs
        # its round-0 parent; gate H0Accepted llr -2.97). The sound reanalyze
        # form needs the GENERATOR net's frozen logits (store child_gen_logit in
        # shards at datagen time). Kept only as evidence; do not use for rounds.
        print("WARNING: live-logit mode is UNSOUND (round-1 postmortem) — training anyway for A/B use only")

    paths = shard_paths(corpus_dir)
    assert paths, f"no shards in {corpus_dir}"
    hold_paths = paths[HOLDOUT_EVERY - 1 :: HOLDOUT_EVERY]
    train_paths = [p for p in paths if p not in set(hold_paths)]
    print(f"corpus: {len(paths)} shards ({len(train_paths)} train / {len(hold_paths)} holdout)")

    # Target sharpness read on the first shard (the calibration sanity gate).
    s0 = read_shard(paths[0])
    effs = np.array([n_eff(t) for t in shard_targets(s0)])
    print(f"targets: N_eff median={np.median(effs):.2f} (band [2.5,6])")

    import os
    device = torch.device(os.environ.get("TETRNN_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))
    model = TetrNet().to(device)
    t0 = time.time()
    mean, std = whitening_stats(train_paths)
    model.feat_mean.copy_(torch.as_tensor(mean))
    model.feat_std.copy_(torch.as_tensor(std))
    print(f"whitening from train children [{time.time()-t0:.0f}s]")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train()
        te = time.time()
        tr: dict[str, list[float]] = {}
        shard_order = np.array(train_paths)
        rng.shuffle(shard_order)
        for sp in shard_order:
            shard = read_shard(str(sp))
            targets = shard_targets(shard)
            for b in shard_batches(shard, rng):
                loss, m = run_batch(
                    model, shard, targets, b, device, metrics=False, live_logits=live
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                for k, v in m.items():
                    tr.setdefault(k, []).append(v)
        trm = {k: float(np.mean(v)) for k, v in tr.items()}

        model.eval()
        ho: dict[str, list[float]] = {}
        with torch.no_grad():
            for sp in hold_paths:
                shard = read_shard(sp)
                targets = shard_targets(shard)
                for b in shard_batches(shard, None):
                    _, m = run_batch(
                        model, shard, targets, b, device, metrics=True, live_logits=live
                    )
                    for k, v in m.items():
                        ho.setdefault(k, []).append(v)
        hom = {k: float(np.mean(v)) for k, v in ho.items()}
        print(
            f"epoch {epoch}: train pCE={trm['policy_ce']:.3f} vCE={trm['value_ce']:.3f} "
            f"sCE={trm.get('slot_ce', 0.0):.3f} | holdout pCE={hom['policy_ce']:.3f} "
            f"vCE={hom['value_ce']:.3f} sCE={hom.get('slot_ce', 0.0):.3f} "
            f"top1={hom['top1']:.3f} z_std={hom['z_hat_std']:.3f} [{time.time()-te:.0f}s]",
            flush=True,
        )
        # Export every epoch: a partial run still yields a loadable model.
        export(model.to("cpu"), out_dir)
        model.to(device)
        print(f"exported epoch {epoch} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
