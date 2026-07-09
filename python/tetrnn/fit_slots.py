"""Fit ONLY the slot head, trunk frozen — the slot-vehicle rehab experiment.

The v3 slot head (trained ~2 epochs as an afterthought term) ranks barely
above random (held-out hit@12 0.34 vs 0.20) and COLLAPSES play when it
filters every beam node (the 2026-07-09 instrument forensic). Because the
head is a single Linear(TRUNK -> N_SLOTS) and the trunk is frozen here, the
parent trunk embeddings can be cached ONCE and the head fit to convergence
as a pure linear multinomial model — minutes, not hours, and the exported
net's leaf eval stays BIT-IDENTICAL (only the filter changes). This also
measures the linear ceiling exactly: if hit@12 plateaus low, the head needs
capacity (net.rs change) or joint fine-tuning, and we know without guessing.

Targets are the same completed-Q pi' as train.py, scattered into slot space
(collided slots sum). Holdout = the lexicographically-last shards.

Usage: uv run python -m tetrnn.fit_slots <corpus-dir> <model-dir> <out-dir>
       [--epochs 200] [--holdout 24]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .export_onnx import load_model
from .shards import read_shard, shard_paths, unpack_plane
from .targets import completed_q_target

N_SLOTS = 104


def cache_embeddings(model, files: list[str]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """One trunk pass per parent decision -> (emb [D, TRUNK], slot_target [D, N_SLOTS], best_slot [D])."""
    embs, targets, bests = [], [], []
    for f in files:
        sh = read_shard(f)
        if sh.parent_own is None or sh.child_slot is None:
            continue
        with torch.no_grad():
            emb = model.trunk(
                torch.as_tensor(unpack_plane(sh.parent_own)).unsqueeze(1).float(),
                torch.as_tensor(unpack_plane(sh.opp_plane)).unsqueeze(1).float(),
                torch.as_tensor(sh.parent_feats).float(),
            )
        st = np.zeros((sh.n_decisions, N_SLOTS), dtype=np.float32)
        bs = np.zeros(sh.n_decisions, dtype=np.int64)
        for k in range(sh.n_decisions):
            c = sh.children_of(k)
            t = completed_q_target(sh.child_score[c].astype(np.float64))
            slots = sh.child_slot[c]
            np.add.at(st[k], slots, t.astype(np.float32))
            bs[k] = slots[np.argmax(t)]
        embs.append(emb)
        targets.append(torch.as_tensor(st))
        bests.append(bs)
    return torch.cat(embs), torch.cat(targets), np.concatenate(bests)


def hit_at(logits: torch.Tensor, best: np.ndarray, k: int) -> float:
    top = torch.topk(logits, k, dim=1).indices.numpy()
    return float(np.mean([b in row for b, row in zip(best, top)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--holdout", type=int, default=24, help="last-N shards held out")
    args = ap.parse_args()

    model = load_model(Path(args.model_dir))
    files = shard_paths(args.corpus)
    tr_files, ho_files = files[: -args.holdout], files[-args.holdout :]
    print(f"caching trunk embeddings: {len(tr_files)} train / {len(ho_files)} holdout shards")
    tr_emb, tr_t, _ = cache_embeddings(model, tr_files)
    ho_emb, ho_t, ho_best = cache_embeddings(model, ho_files)
    print(f"decisions: train {len(tr_emb)}, holdout {len(ho_emb)}")

    head = torch.nn.Linear(tr_emb.shape[1], N_SLOTS)
    head.load_state_dict(model.slot_head.state_dict())  # warm start from v3
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    n = len(tr_emb)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for lo in range(0, n, 4096):
            idx = perm[lo : lo + 4096]
            loss = -(tr_t[idx] * torch.log_softmax(head(tr_emb[idx]), dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if epoch % 10 == 9 or epoch == 0:
            with torch.no_grad():
                ho_logits = head(ho_emb)
                ho_ce = -(ho_t * torch.log_softmax(ho_logits, dim=1)).sum(1).mean()
            print(
                f"epoch {epoch + 1}: train sCE {tot / n:.3f} | holdout sCE {float(ho_ce):.3f} "
                f"hit@12 {hit_at(ho_logits, ho_best, 12):.3f} hit@24 {hit_at(ho_logits, ho_best, 24):.3f}",
                flush=True,
            )

    # Export: copy the model dir, overwrite only the slot head tensors.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("config.json",):
        shutil.copy(Path(args.model_dir) / f, out / f)
    sd = load_file(str(Path(args.model_dir) / "net_v2.safetensors"))
    sd["slot_head.weight"] = head.weight.detach().contiguous()
    sd["slot_head.bias"] = head.bias.detach().contiguous()
    save_file(sd, str(out / "net_v2.safetensors"))
    meta = {"fit_slots": {"corpus": args.corpus, "base": args.model_dir, "epochs": args.epochs}}
    (out / "fit_slots.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {out} (leaf eval bit-identical to base; slot head refit)")


if __name__ == "__main__":
    main()
