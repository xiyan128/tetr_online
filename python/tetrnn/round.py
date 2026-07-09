"""One resumable command per expert-iteration round (leapfrog T16).

Encodes the campaign's burned-in lessons: consistent vehicle end-to-end (the
guided beam drives datagen AND the gates), a DIVERSIFIED self-play pool (half
grounded net-vs-CC2, half mirror — homogeneous pools bred a parent-exploiting
degenerate, A-r8), fine-tune from the LINEAGE net (from-scratch regresses;
lineage chains from the newest net while the INCUMBENT advances only on
promotion, A-r7), SSL aux on, static completed-Q targets (live-logit
quarantined), and — load-bearing — promotion requires the incumbent gate PASS
**and** no-regression vs the fixed CC2 anchor (self-play lineages game
incumbent-only gates, A-r8).

Steps (each skipped if its output already exists — rerun == resume):
  1. datagen   — half `--opp-cc2` grounded + half mirror, per-mode subdirs
  2. mix       — replay symlinks: round shards + every 4th base-corpus shard
  3. train     — fine-tune from the lineage net (--init, --ssl), 1 epoch
  4. duels     — policy/value isolation + the CC2 ANCHOR duel (telemetry+veto)
  5. gate      — latched pair-GSPRT guided-vs-guided vs the incumbent
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
    ap.add_argument("--incumbent", required=True, help="gate opponent (advances only on PASS)")
    ap.add_argument("--lineage", default=None, help="training/datagen net (default: incumbent). A-r7: lineage chains from the newest net (AZ-standard); the incumbent advances only on gate PASS.")
    ap.add_argument("--base-corpus", required=True, help="grounded base corpus for the replay mix")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--topm", type=int, default=12)
    ap.add_argument("--wd", default="w8d5")
    ap.add_argument("--lr", default=None, help="fine-tune LR override (A-r10: 1e-3 rewrites the policy wholesale in one epoch — every lineage variant became an anchor-failing incumbent-exploiter; 1e-4 is the small-delta regime)")
    ap.add_argument("--vehicle", choices=["guided", "sguided"], default="guided", help="EXPLICIT vehicle, end-to-end (datagen + duels + gate): guided = per-child ranker (validated, slow), sguided = slot ranker (fast; qualify hit@12 + anchor first). The hidden chooser is how rounds 6-10 died (0cebd90).")
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
    lineage = args.lineage or args.incumbent
    row: dict = {"round": n, "incumbent": args.incumbent, "lineage": lineage, "games": args.games,
                 "vehicle": args.vehicle}

    # 1. datagen — A-r8 pool diversity: half grounded (vs CC2), half mirror.
    # A homogeneous pool let the lineage evolve a parent-exploiting degenerate.
    if not (corpus / "cc2").exists():
        half = args.games // 2
        for tag, extra, base in [
            ("cc2", ["--opp-cc2"], datagen_seeds),
            ("mirror", [], datagen_seeds + half),
        ]:
            text = sh(
                [
                    str(BIN), "datagen",
                    "--net", lineage,
                    "--topm", str(args.topm),
                    "--width", width, "--depth", depth,
                    "--games", str(half),
                    "--seeds", str(base),
                    "--workers", str(args.workers),
                    *(["--slot-vehicle"] if args.vehicle == "sguided" else []),
                    *extra,
                    "--out", str(corpus / tag),
                ],
                rdir / f"datagen_{tag}.log",
            )
            row[f"datagen_{tag}"] = last_json(text)
    else:
        print(f"datagen: {corpus} exists — skipping")

    # 2. replay mix: this round's shards + every 4th base-corpus shard.
    if not mix.exists():
        mix.mkdir()
        k = 0
        for f in sorted(corpus.glob("*/w*/shard-*.safetensors")):
            tag = f.parent.parent.name
            (mix / f"shard-r{n}{tag}{f.parent.name}-{f.name.removeprefix('shard-')}").symlink_to(f)
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
                f"--init={lineage}", "--ssl",
                *([f"--lr={args.lr}"] if args.lr else []),
            ],
            rdir / "train.log",
        )
    else:
        print(f"train: {net} exists — skipping")
    row["train_tail"] = (rdir / "train.log").read_text().strip().splitlines()[-3:] if (rdir / "train.log").exists() else []

    # 4. isolation duels (telemetry, not verdicts).
    cand_guided = f"{args.vehicle}:{net}@m{args.topm}{args.wd}"
    inc_guided = f"{args.vehicle}:{args.incumbent}@m{args.topm}{args.wd}"
    for tag, a, b, seeds in [
        ("policy_duel", f"policy:{net}", f"policy:{args.incumbent}", duel_seeds),
        ("value_duel", f"value:{net}", f"value:{args.incumbent}", duel_seeds + 100_000),
        # A-r8: the FIXED-ANCHOR duel — self-play lineages can evolve
        # parent-exploiting degenerates (r7: beat v3 at the gate, lost 0-32 to
        # CC2). Promotion requires no-regression vs the fixed external anchor.
        ("anchor_duel", cand_guided, "beam:cc2@w8d5", duel_seeds + 200_000),
    ]:
        text = sh(
            [str(BIN), "duel", "--a", a, "--b", b, "--pairs", "24",
             "--seeds", str(seeds), "--allow-dirty"],
            rdir / f"{tag}.log",
        )
        row[tag] = duel_line(text)
        print(f"{tag}: {row[tag]}")

    # 5. promotion gate (the verdict) — SHORT-CIRCUITED when the anchor
    # already failed (A-r8 veto): rounds 6-9 each burned 5-35 gate-minutes on
    # candidates the 25-second anchor duel had already killed.
    anchor_wins = int(row["anchor_duel"].split("|")[1].strip().split()[1].split("-")[0])
    row["anchor_wins_of_48"] = anchor_wins
    if anchor_wins < 18:
        row["gate"] = "skipped (anchor veto)"
        row["verdict"] = f"ANCHOR_FAIL({anchor_wins}/48)"
    else:
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
