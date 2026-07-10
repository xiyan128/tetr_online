"""One command per expert-iteration round.

    datagen -> train (this round + replay of earlier rounds) -> gate

A candidate is kept only if it (a) beats the incumbent in a seed-paired duel
AND (b) does not regress against the fixed CC2 anchor (`beam:cc2@w8d5`).
Round 0 has no incumbent: its corpus comes from CC2 self-play (the only CC2
supervision in the campaign) and its net is accepted on the anchor read alone.

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
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--incumbent", default=None, help="current best model dir (omit for round 0)")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wd", default="w8d5", help="beam width/depth for datagen and duels")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=24, help="seed pairs per duel")
    ap.add_argument("--seed-base", type=int, default=10_000_000)
    ap.add_argument("--duel-base", type=int, default=900_000_000)
    args = ap.parse_args()

    n = args.round
    scratch = Path(args.scratch)
    rdir = scratch / f"r{n}"
    rdir.mkdir(parents=True, exist_ok=True)
    corpus = rdir / "corpus"
    net = rdir / "net"
    width, depth = args.wd[1:].split("d")
    t0 = time.time()
    row: dict = {"round": n, "incumbent": args.incumbent, "games": args.games, "wd": args.wd}
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

    # 2. train: this round's corpus + a replay of every earlier round's.
    if (net / "config.json").exists():
        print(f"train: {net} exists — skipping")
    else:
        replay = [
            str(scratch / f"r{k}" / "corpus")
            for k in range(n)
            if (scratch / f"r{k}" / "corpus" / "receipt.json").exists()
        ]
        sh(
            [
                "uv", "run", "--directory", str(REPO / "python"), "python", "-m",
                "tetrnn.train", str(corpus), *replay, str(net),
                "--epochs", str(args.epochs),
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
        text = sh(
            [str(BIN), "duel", "--a", cand, "--b", f"beam:{args.incumbent}@{args.wd}",
             "--pairs", str(args.pairs), "--seeds", str(duel_seeds), "--allow-dirty"],
            rdir / "duel_incumbent.log",
        )  # fmt: skip
        row["vs_incumbent"] = duel_summary(text)
        verdicts.append(wins_of(row["vs_incumbent"]) > args.pairs)  # majority of 2*pairs games
        print(f"vs incumbent: {row['vs_incumbent']}")
    text = sh(
        [str(BIN), "duel", "--a", cand, "--b", "beam:cc2@w8d5",
         "--pairs", str(args.pairs), "--seeds", str(duel_seeds + 500_000), "--allow-dirty"],
        rdir / "duel_anchor.log",
    )  # fmt: skip
    row["vs_anchor"] = duel_summary(text)
    print(f"vs anchor: {row['vs_anchor']}")
    anchor_wins = wins_of(row["vs_anchor"])
    row["anchor_wins"] = anchor_wins
    # No-regression bar: a third of the anchor games (round 0 calibrates this).
    verdicts.append(anchor_wins * 3 >= args.pairs * 2)
    row["verdict"] = "PROMOTE" if all(verdicts) else "KEEP_INCUMBENT"
    row["wall_secs"] = round(time.time() - t0, 1)

    with (scratch / "rounds.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nROUND {n}: {row['verdict']}  (anchor {row['vs_anchor']})  [{row['wall_secs']}s]")


if __name__ == "__main__":
    main()
