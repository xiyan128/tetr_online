"""One resumable command per expert-iteration round (leapfrog T16).

Encodes the pipeline proven by hand in rounds 1-5, with the lessons burned in:
consistent vehicle end-to-end (the guided slot beam drives datagen AND the
gates), grounded two-arm data (net vs CC2 — mirror self-play z is weakly
predictable), fine-tune from the incumbent (from-scratch regresses), SSL aux
on (the best value read), static completed-Q targets (live-logit quarantined),
first-tie argmax + seat parity (driver == harness, seed-matched).

Steps (each skipped if its output already exists — rerun == resume):
  1. datagen   — two-arm grounded games via `tetr-research datagen --opp-cc2`
  2. mix       — replay symlinks: round shards + every 4th incumbent-corpus shard
  3. train     — fine-tune from the incumbent (--init, --ssl), 1 epoch
  4. duels     — policy/value isolation reads vs the incumbent (telemetry)
  5. gate      — latched pair-GSPRT guided-vs-guided (p1=0.55): the verdict
  6. ledger    — one JSON line appended to <scratch>/rounds.jsonl

Seed regions (disjoint by construction, all logged):
  datagen: 3_000_000 + round * 100_000        (games consume +games)
  duels:   993_000_000 + round * 1_000_000
  gate:    994_000_000 + round * 1_000_000

Usage:
  uv run python -m tetrnn.round --round N --incumbent <model-dir> \
      --base-corpus <round0-corpus-dir> --scratch <dir> [--games 1200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "target/release/tetr-research"


def sh(cmd: list[str], log: Path | None = None) -> str:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    if log:
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


def duel_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("duel |") or line.startswith("gate |"):
            return line.strip()
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--incumbent", required=True, help="model dir (e.g. round0_v3)")
    ap.add_argument("--base-corpus", required=True, help="grounded base corpus for the replay mix")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--topm", type=int, default=12)
    ap.add_argument("--wd", default="w8d5")
    args = ap.parse_args()

    n = args.round
    scratch = Path(args.scratch)
    rdir = scratch / f"r{n}"
    rdir.mkdir(parents=True, exist_ok=True)
    corpus = rdir / "corpus"
    mix = rdir / "mix"
    net = rdir / "net"
    ledger = scratch / "rounds.jsonl"
    width, depth = args.wd[1:].split("d")
    datagen_seeds = 3_000_000 + n * 100_000
    duel_seeds = 993_000_000 + n * 1_000_000
    gate_seeds = 994_000_000 + n * 1_000_000
    t0 = time.time()
    row: dict = {"round": n, "incumbent": args.incumbent, "games": args.games}

    # 1. datagen (grounded two-arm, consistent vehicle).
    if not (corpus / "w0").exists():
        text = sh(
            [
                str(BIN), "datagen",
                "--net", args.incumbent,
                "--topm", str(args.topm),
                "--width", width, "--depth", depth,
                "--games", str(args.games),
                "--seeds", str(datagen_seeds),
                "--workers", str(args.workers),
                "--opp-cc2",
                "--out", str(corpus),
            ],
            rdir / "datagen.log",
        )
        row["datagen"] = last_json(text)
    else:
        print(f"datagen: {corpus} exists — skipping")

    # 2. replay mix: this round's shards + every 4th base-corpus shard.
    if not mix.exists():
        mix.mkdir()
        k = 0
        for f in sorted(corpus.glob("w*/shard-*.safetensors")):
            (mix / f"shard-r{n}{f.parent.name}-{f.name.removeprefix('shard-')}").symlink_to(f)
        for i, f in enumerate(sorted(Path(args.base_corpus).glob("**/shard-*.safetensors"))):
            if i % 4 == 0:
                (mix / f"shard-base-{f.name.removeprefix('shard-')}").symlink_to(f)
                k += 1
        print(f"mix: {len(list(mix.iterdir()))} shards ({k} base)")

    # 3. train: fine-tune from the incumbent, SSL on, static targets, 1 epoch.
    if not (net / "config.json").exists():
        sh(
            [
                "uv", "run", "--directory", str(REPO / "python"), "python", "-m",
                "tetrnn.train", str(mix), str(net), "1",
                f"--init={args.incumbent}", "--ssl",
            ],
            rdir / "train.log",
        )
    else:
        print(f"train: {net} exists — skipping")
    row["train_tail"] = (rdir / "train.log").read_text().strip().splitlines()[-3:] if (rdir / "train.log").exists() else []

    # 4. isolation duels (telemetry, not verdicts).
    cand_guided = f"guided:{net}@m{args.topm}{args.wd}"
    inc_guided = f"guided:{args.incumbent}@m{args.topm}{args.wd}"
    for tag, a, b, seeds in [
        ("policy_duel", f"policy:{net}", f"policy:{args.incumbent}", duel_seeds),
        ("value_duel", f"value:{net}", f"value:{args.incumbent}", duel_seeds + 100_000),
    ]:
        text = sh(
            [str(BIN), "duel", "--a", a, "--b", b, "--pairs", "24",
             "--seeds", str(seeds), "--allow-dirty"],
            rdir / f"{tag}.log",
        )
        row[tag] = duel_line(text)
        print(f"{tag}: {row[tag]}")

    # 5. promotion gate (the verdict).
    text = sh(
        [str(BIN), "gate", "--a", cand_guided, "--b", inc_guided,
         "--seeds", str(gate_seeds), "--max-pairs", "120", "--allow-dirty"],
        rdir / "gate.log",
    )
    row["gate"] = duel_line(text)
    row["gate_json"] = last_json(text)
    row["verdict"] = row["gate_json"].get("verdict")
    row["wall_secs"] = round(time.time() - t0, 1)

    # 6. ledger.
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nROUND {n}: {row['verdict']}  ({row['gate']})  [{row['wall_secs']}s]")
    print(f"ledger: {ledger}")


if __name__ == "__main__":
    main()
