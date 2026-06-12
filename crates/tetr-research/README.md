# tetr-research

The deterministic experiment platform behind the versus bot. **Bevy-free**: it
drives `tetr-core`'s engine + AI directly, so every result is a pure function
of `(commit, eval, bots…)` — three names, all stamped into a run receipt.
Always run `--release` (a debug match is ~20× slower).

```sh
cargo run --release -p tetr-research -- run marathon dt20
cargo run --release -p tetr-research -- run versus cc2-default dt20
cargo run --release -p tetr-research -- run race probe-tp128d9 attack-tuned-d3
duckdb -init scripts/research.sql      # analyze every run ever recorded
```

## How it fits together

- **Two registries, read as code** (there is no `list`): bots in
  [`src/bots.rs`](src/bots.rs) (named `BotSpec`s — search × eval × sight),
  evals in [`src/registry.rs`](src/registry.rs) (named, typed measurement
  specs). The binary pairs them at the prompt: `run <eval> [bots…]`. Anything
  result-affecting is a new registered name, never a flag; the only flags are
  machine-local (`--budget-secs`, `--cc2-bin`, `--runs-root`, `--allow-dirty`).
- **Runs refuse a dirty tree** by default (such runs aren't re-runnable from
  names); `--allow-dirty` records an exploratory run, stamped in its receipt.
- **One JSON line on stdout** per run (`{run, eval, bots, …headline metrics}`);
  humans read stderr. Each run directory holds `spec.json` (the reproducibility
  receipt) and `games.jsonl` (normalized per-game facts) — analysis happens in
  duckdb, never in the platform. The invariants are an ADR:
  [`docs/adr-data-architecture.md`](../../docs/adr-data-architecture.md).
- **Evals**: `marathon` (score/sec + APP), `downstack` (censored cheese
  pieces), `versus` (arm-swapped head-to-head, deaths first-class), `race`
  (pair-GSPRT survival verdict), `marathon-holdout[-long]` (one read per
  candidate), `cc2-baseline-*` (real Cold Clear 2 over TBP), and the
  optimizer `app-climb` (a (1+1)-ES over a subject's Cc2 weight surface,
  campaign-seeded, self-validating).
- **Long jobs self-bound**: optimizers and races honour `--budget-secs` and
  exit with an honest partial verdict. A budget-cut climb is a prefix of the
  unbounded walk; pin its `iters` to reproduce a full stream byte-for-byte.

The task-oriented guide is [`docs/research-guide.md`](../../docs/research-guide.md);
the binding rules (determinism, seed regions, arm-swap + CRN, death decides)
live in the crate docs ([`src/lib.rs`](src/lib.rs)). Results worth keeping are
RUN RECORDs in the command doc headers, citing run ids.

## Current SOTA snapshot (2026-06-12)

The APP campaign's champion is **`probe-tp128d9`** — a transposition-pruned
beam (width 128, depth 9, opt-in `BeamPlanner::transposing`; plain beams stay
byte-identical to their recorded baselines) over the unchanged `attack_tuned`
CC2 evaluator:

- **0.8225 APP held-out** (0.8178 at cap 600 — no opening artifact), versus
  the shipped depth-2 bot's 0.46 and the old ~0.67 era ceiling;
- **downstack 17.17** censored pieces (beats attack-tuned-d3's 18.67);
- **race vs attack-tuned-d3: H1, 63-0-1** (LLR +3.34).

Every gain came from **search class** (depth → width → best-first → TP-beam);
every eval-side lever was null — the `app-climb` run record and the `probe-*`
registrations in `bots.rs` carry the receipts. Attack per line rises with
search quality (1.55 → 2.05 at flat lines/game): concentrated B2B/T-spin
attack, not the combo-farm that the empty-board APP metric can be gamed by.
Past ~0.83 the recorded map says RL/self-play, not tuning. None of this is
shipped into the game yet — that is a separate latency decision.
