"""The value trainer: win/draw/loss cross-entropy on played states.

Each shard row is one played state + the game outcome from the mover's
perspective; the loss is plain 3-class CE. Whitening stats come from the
training rows (one row role — every row is served the same way). Holdout is
every 10th shard; shard flushes are game-aligned, so that is a game-level
split.

`--td ALPHA` switches to TD-style bootstrapped targets (the TD-Gammon move):
each row's target becomes `alpha * frozen_probs(next state) + (1-alpha) *
onehot(z)`, with a seat's LAST row always hard-labeled z. The frozen copy
refreshes each epoch. Rationale (2026-07-11): pure-outcome labels measurably
carry no mid-game signal in balanced games and only across-game signal in
unbalanced ones — a beam needs within-decision discrimination, which
bootstrapping propagates from the grounded terminals. Rows are consecutive
per (game, seat, ply) inside a shard, so TD needs no schema change.

Usage:
  uv run python -m tetrnn.train <corpus-dir> [<corpus-dir> ...] <out-model-dir>
      [--epochs 3] [--init <model-dir>] [--lr 1e-3] [--td 0.5] [--rank 1.0]

`--rank W` adds a pairwise ranking loss over the stored (played, sibling)
pairs: the search preferred the played placement, so z_hat(played) should
exceed z_hat(sibling). Unit-free within-decision supervision — the signal
outcome labels measurably lack.

Multiple corpus dirs concatenate (the replay-buffer form: pass the current
round's corpus plus earlier rounds').
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from .export import export, load
from .model import TetrNet
from .shards import read_shard, shard_paths, unpack_plane

BATCH = 512
HOLDOUT_EVERY = 10  # every 10th shard is holdout (game-aligned => game-level)


def z_class(z: np.ndarray) -> np.ndarray:
    """z=+1 -> win(0), z=0 -> draw(1), z=-1 -> loss(2) (net.rs head order)."""
    return (1 - z).astype(np.int64)


def whitening_stats(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std over every training row's features, one streaming pass."""
    n, s, s2 = 0, None, None
    for p in paths:
        f = read_shard(p).feats.astype(np.float64)
        n += f.shape[0]
        s = f.sum(axis=0) if s is None else s + f.sum(axis=0)
        s2 = (f * f).sum(axis=0) if s2 is None else s2 + (f * f).sum(axis=0)
    assert s is not None and s2 is not None, "no training rows"
    mean = s / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def successor_of(shard) -> np.ndarray:
    """`succ[i]` = row index of the same seat's next decision in the same game,
    or -1 (terminal for that seat). Rows are written in play order, so the next
    row with the same (game_id, seat) is ply+1."""
    succ = np.full(shard.n_rows, -1, dtype=np.int64)
    last: dict[tuple[int, int], int] = {}
    for i in range(shard.n_rows):
        key = (int(shard.decision[i, 0]), int(shard.decision[i, 1]))
        if key in last:
            succ[last[key]] = i
        last[key] = i
    return succ


def epoch_pass(
    model: TetrNet,
    paths: list[str],
    device: torch.device,
    opt: torch.optim.Optimizer | None,
    rng: np.random.Generator | None,
    td: float = 0.0,
    frozen: TetrNet | None = None,
    rank: float = 0.0,
) -> tuple[float, float]:
    """One pass over `paths` (shard-streamed). With `opt` it trains; without it
    it evaluates (always against the plain z labels — the grounded metric).
    Returns (mean CE, accuracy of the WDL argmax vs z)."""
    total_ce, total_hit, total_n = 0.0, 0, 0
    for p in paths:
        shard = read_shard(p)
        succ = successor_of(shard) if td > 0 and frozen is not None else None
        order = np.arange(shard.n_rows)
        if rng is not None:
            rng.shuffle(order)
        for lo in range(0, len(order), BATCH):
            rows = order[lo : lo + BATCH]
            board = torch.as_tensor(unpack_plane(shard.own[rows]), device=device).unsqueeze(1)
            feats = torch.as_tensor(shard.feats[rows], device=device)
            target = torch.as_tensor(z_class(shard.z[rows]), device=device)
            if opt is None:
                with torch.no_grad():
                    logits = model.serve(board, feats)
                    ce = torch.nn.functional.cross_entropy(logits, target)
            else:
                logits = model.serve(board, feats)
                rank_loss = torch.zeros((), device=device)
                if rank > 0:
                    # Pairwise ranking: the search preferred `own` over `alt`.
                    # z_hat difference through a logistic loss — unit-free
                    # within-decision supervision (outcome labels can't rank
                    # siblings; absolute scores carry generator units).
                    live = shard.has_alt[rows] > 0
                    if live.any():
                        ab = torch.as_tensor(
                            unpack_plane(shard.alt_own[rows[live]]), device=device
                        ).unsqueeze(1)
                        af = torch.as_tensor(shard.alt_feats[rows[live]], device=device)
                        alt_logits = model.serve(ab, af)
                        zh = lambda lg: (  # noqa: E731 — p_win − p_loss
                            torch.softmax(lg, dim=1)[:, 0] - torch.softmax(lg, dim=1)[:, 2]
                        )
                        mask = torch.as_tensor(live, device=device)
                        gap = zh(logits[mask]) - zh(alt_logits)
                        rank_loss = -torch.nn.functional.logsigmoid(5.0 * gap).mean()
                if succ is not None and frozen is not None:
                    # TD target: soft successor belief mixed with the outcome;
                    # terminal rows stay hard-grounded at z.
                    soft = torch.nn.functional.one_hot(target, 3).float()
                    nxt = succ[rows]
                    live = nxt >= 0
                    if live.any():
                        nb = torch.as_tensor(
                            unpack_plane(shard.own[nxt[live]]), device=device
                        ).unsqueeze(1)
                        nf = torch.as_tensor(shard.feats[nxt[live]], device=device)
                        with torch.no_grad():
                            probs = torch.softmax(frozen.serve(nb, nf), dim=1)
                        mask = torch.as_tensor(live, device=device)
                        soft[mask] = td * probs + (1 - td) * soft[mask]
                    ce = -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                else:
                    ce = torch.nn.functional.cross_entropy(logits, target)
                total = ce if rank <= 0 else ce + rank * rank_loss
                opt.zero_grad()
                total.backward()
                opt.step()
            total_ce += float(ce.detach()) * len(rows)
            total_hit += int((logits.argmax(dim=1) == target).sum())
            total_n += len(rows)
    return total_ce / max(total_n, 1), total_hit / max(total_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="corpus dir(s), then the output model dir last")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--init", default=None, help="fine-tune from this exported model dir")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--td",
        type=float,
        default=0.0,
        help="bootstrap weight: target = td*frozen_probs(next) + (1-td)*onehot(z); "
        "0 = plain outcome CE",
    )
    ap.add_argument(
        "--rank",
        type=float,
        default=0.0,
        help="weight of the pairwise ranking loss (search preferred played over "
        "the stored sibling); 0 = off",
    )
    args = ap.parse_args()
    *corpora, out_dir = args.dirs
    out = Path(out_dir)

    paths = [p for d in corpora for p in shard_paths(d)]
    if not paths:
        raise SystemExit(f"no shards under {corpora}")
    train_paths = [p for i, p in enumerate(paths) if i % HOLDOUT_EVERY != 0]
    holdout_paths = [p for i, p in enumerate(paths) if i % HOLDOUT_EVERY == 0]

    device = torch.device(
        os.environ.get("TETRNN_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    )
    t0 = time.time()
    if args.init:
        model = load(Path(args.init))
        print(f"init from {args.init} (whitening kept)")
    else:
        model = TetrNet()
        mean, std = whitening_stats(train_paths)
        model.feat_mean.copy_(torch.as_tensor(mean))
        model.feat_std.copy_(torch.as_tensor(std))
        print(f"whitening from {len(train_paths)} train shards [{time.time() - t0:.0f}s]")
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    print(
        f"corpus: {len(train_paths)} train / {len(holdout_paths)} holdout shards, "
        f"lr={args.lr}, td={args.td}, rank={args.rank}"
    )
    for epoch in range(args.epochs):
        frozen = None
        if args.td > 0:
            # The bootstrap source: last epoch's export (or the init weights on
            # epoch 0 — near-uniform beliefs, so early TD ≈ label smoothing).
            import copy

            frozen = copy.deepcopy(model).eval()
            for q in frozen.parameters():
                q.requires_grad_(False)
        tr_ce, tr_acc = epoch_pass(model, train_paths, device, opt, rng, args.td, frozen, args.rank)
        model.eval()
        ho_ce, ho_acc = epoch_pass(model, holdout_paths, device, None, None)
        model.train()
        # Export every epoch: a partial run still yields a loadable model.
        export(model.to("cpu").eval(), out)
        model.to(device).train()
        print(
            f"epoch {epoch}: train CE={tr_ce:.4f} acc={tr_acc:.3f} | "
            f"holdout CE={ho_ce:.4f} acc={ho_acc:.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    print(f"exported {out}")


if __name__ == "__main__":
    main()
