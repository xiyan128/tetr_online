"""One command per expert-iteration round.

    datagen -> train (this round + replay of earlier rounds) -> gate

A candidate is kept only if it (a) beats the incumbent in a seed-paired duel
AND (b) clears a fixed floor against the CC2 anchor (`beam:cc2@w8d5`).
Round 0 has no incumbent: its corpus comes from CC2 self-play (the only CC2
supervision in the campaign), its anchor duel is recorded as the BASELINE
(it calibrates the floor), and its net becomes the first incumbent.

Prerequisite: `cargo build --release -p tetr-research` (the driver shells out
to that binary). Run from anywhere; paths are absolute.

Seeds: datagen uses `--seed-base + round * 1_000_000` (games consume one seed
each); duels use a disjoint region derived the same way from `--duel-base`.
Every step's receipt lands in `<scratch>/rN/` and one JSON line per round is
appended to `<scratch>/rounds.jsonl`. Steps re-run from scratch if their
output is missing; a completed datagen is detected by its receipt file, never
by directory existence (a killed run leaves partial shards).

Usage:
  uv run python -m tetrnn.round --round N --scratch <dir>
      [--incumbent <model-dir>] [--games 600] [--wd w8d5] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "target/release/tetr-research"


def sh(cmd: list[str], log: Path) -> str:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    log.write_text(text)
    if out.returncode != 0:
        print(text[-2000:], file=sys.stderr)
        raise SystemExit(f"step failed rc={out.returncode}: {cmd[0]}")
    return text


def last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit("no JSON receipt in output")


def duel_summary(text: str) -> str:
    for line in text.splitlines():
        if line.startswith(("duel |", "gate |")):
            return line.strip()
    return "?"


def wins_of(summary: str) -> int:
    """'duel | A 11-5-0 over 16 games | ...' -> 11."""
    return int(summary.split("|")[1].strip().split()[1].split("-")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, help="round number; 0 bootstraps from CC2")
    ap.add_argument(
        "--scratch", required=True, help="campaign dir: rN/ per round + rounds.jsonl ledger"
    )
    ap.add_argument("--incumbent", default=None, help="current best model dir (omit for round 0)")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wd", default="w8d5", help="beam width/depth for datagen and duels")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=24, help="seed pairs per duel")
    ap.add_argument("--seed-base", type=int, default=10_000_000)
    ap.add_argument("--duel-base", type=int, default=900_000_000)
    ap.add_argument(
        "--rank",
        type=float,
        default=1.0,
        help="pairwise ranking-loss weight (the standard recipe since the "
        "2026-07-11 breakthrough: outcome CE alone cannot rank sibling "
        "placements; rank pairs took the anchor from 0-48 to 24-24)",
    )
    ap.add_argument(
        "--finetune",
        action="store_true",
        help="train --init from the incumbent instead of from scratch "
        "(round 1's from-scratch-on-mix candidate lost 6-42 to its incumbent)",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="let duels run on a dirty tree (receipts get stamped dirty; "
        "results are then NOT reproducible from (commit, seed))",
    )
    ap.add_argument(
        "--ground-cap",
        type=int,
        default=0,
        help="cap the round-0 bootstrap corpus in the replay to this many shards "
        "(even stride), keeping later self-play rounds full — rebalances the "
        "replay so the ~600-game self-play isn't drowned by the 20k CC2 base "
        "(0 = uncapped; round 2 plateaued at the 33:1 imbalance)",
    )
    args = ap.parse_args()

    n = args.round
    scratch = Path(args.scratch)
    rdir = scratch / f"r{n}"
    rdir.mkdir(parents=True, exist_ok=True)
    corpus = rdir / "corpus"
    net = rdir / "net"
    width, depth = args.wd[1:].split("d")
    t0 = time.time()
    row: dict = {
        "round": n,
        "incumbent": args.incumbent,
        "games": args.games,
        "wd": args.wd,
        "finetune": args.finetune,
        "rank": args.rank,
        "ground_cap": args.ground_cap,
    }
    if n > 0 and not args.incumbent:
        raise SystemExit("rounds past 0 need --incumbent (the model whose self-play trains next)")

    # 1. datagen: round 0 = CC2 self-play (the bootstrap teacher); later rounds
    # = the incumbent's self-play. Resume via the receipt, not the dir.
    receipt = corpus / "receipt.json"
    if receipt.exists():
        row["datagen"] = json.loads(receipt.read_text())
        print("datagen: receipt exists — skipping")
    else:
        if corpus.exists():
            print("datagen: partial dir without receipt — regenerating")
            shutil.rmtree(corpus)
        text = sh(
            [
                str(BIN), "datagen",
                *(["--net", args.incumbent] if n > 0 and args.incumbent else []),
                "--width", width, "--depth", depth,
                "--games", str(args.games),
                "--seeds", str(args.seed_base + n * 1_000_000),
                "--workers", str(args.workers),
                "--out", str(corpus),
            ],
            rdir / "datagen.log",
        )  # fmt: skip
        row["datagen"] = last_json(text)
        receipt.write_text(json.dumps(row["datagen"]))

    # 2. train: this round's corpus + a replay of every earlier round's. The
    # round-0 bootstrap (20k CC2 games) is optionally CAPPED so it grounds
    # without drowning the self-play signal — round 2 plateaued at a 33:1
    # replay imbalance (self-play is the only new information each round).
    if (net / "config.json").exists():
        print(f"train: {net} exists — skipping")
    else:
        replay = []
        for k in range(n):
            ck = scratch / f"r{k}" / "corpus"
            if not (ck / "receipt.json").exists():
                continue
            if k == 0 and args.ground_cap > 0:
                shards = sorted(ck.glob("**/shard-*.safetensors"))
                if len(shards) > args.ground_cap:
                    stride = len(shards) / args.ground_cap
                    shards = [shards[int(i * stride)] for i in range(args.ground_cap)]
                # Symlink the strided subset into rN/ground/ (real dir, so the
                # trainer's `**/shard-*.safetensors` glob finds the linked files).
                gdir = rdir / "ground"
                gdir.mkdir(exist_ok=True)
                for i, sp in enumerate(shards):
                    link = gdir / f"shard-{i:05d}.safetensors"
                    if not link.exists():
                        link.symlink_to(sp.resolve())
                replay.append(str(gdir))
                print(f"replay: r0 capped to {len(shards)} shards (ground); r1..r{n - 1} full")
            else:
                replay.append(str(ck))
        init = ["--init", args.incumbent] if args.finetune and args.incumbent else []
        sh(
            [
                "uv", "run", "--directory", str(REPO / "python"), "python", "-m",
                "tetrnn.train", str(corpus), *replay, str(net),
                "--epochs", str(args.epochs), "--rank", str(args.rank), *init,
            ],
            rdir / "train.log",
        )  # fmt: skip
    row["train_tail"] = (
        (rdir / "train.log").read_text().strip().splitlines()[-2:]
        if (rdir / "train.log").exists()
        else []
    )

    # 3. gate: candidate vs incumbent (skipped at round 0) + the CC2 anchor.
    cand = f"beam:{net}@{args.wd}"
    duel_seeds = args.duel_base + n * 1_000_000
    verdicts = []
    if args.incumbent:
        dirty = ["--allow-dirty"] if args.allow_dirty else []
        text = sh(
            [str(BIN), "duel", "--a", cand, "--b", f"beam:{args.incumbent}@{args.wd}",
             "--pairs", str(args.pairs), "--seeds", str(duel_seeds), *dirty],
            rdir / "duel_incumbent.log",
        )  # fmt: skip
        row["vs_incumbent"] = duel_summary(text)
        verdicts.append(wins_of(row["vs_incumbent"]) > args.pairs)  # majority of 2*pairs games
        print(f"vs incumbent: {row['vs_incumbent']}")
    text = sh(
        [str(BIN), "duel", "--a", cand, "--b", "beam:cc2@w8d5",
         "--pairs", str(args.pairs), "--seeds", str(duel_seeds + 500_000),
         *(["--allow-dirty"] if args.allow_dirty else [])],
        rdir / "duel_anchor.log",
    )  # fmt: skip
    row["vs_anchor"] = duel_summary(text)
    print(f"vs anchor: {row['vs_anchor']}")
    anchor_wins = wins_of(row["vs_anchor"])
    row["anchor_wins"] = anchor_wins
    if n == 0:
        # Round 0 has nothing to beat: its anchor read IS the baseline the
        # floor is calibrated against, and its net is the first incumbent.
        row["verdict"] = "BASELINE"
    else:
        # Anchor floor: a third of the anchor games (vs the round-0 baseline).
        verdicts.append(anchor_wins * 3 >= args.pairs * 2)
        row["verdict"] = "PROMOTE" if all(verdicts) else "KEEP_INCUMBENT"
    row["wall_secs"] = round(time.time() - t0, 1)

    with (scratch / "rounds.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nROUND {n}: {row['verdict']}  (anchor {row['vs_anchor']})  [{row['wall_secs']}s]")


if __name__ == "__main__":
    main()
