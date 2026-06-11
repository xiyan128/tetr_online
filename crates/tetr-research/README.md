# tetr-research

Headless metrics, benchmarks, and autoresearch tooling for the Tetris AI. Everything
here is **deterministic** (seeded RNG, no clock in decisions) and **Bevy-free**: it
drives `tetr-core`'s engine + AI directly so results are reproducible and drop straight
into tuning loops. The lib (`lib.rs`) holds the bot specs, the versus/marathon
harnesses, and `decide_versus`; `behavior.rs` is the attack-metrics suite.

## Primary metric

**APP** (attack per piece), measured across garbage **scenarios** (`standard_suite()`):
`clean`, `cheese9`, `faucet1/4`, `faucet1/2`. Also reported: DS/P (downstack/piece),
survival, attack/line, and **`ms/piece`**, the compute axis of the compute/quality
frontier. `ms/piece` is only meaningful on an **unloaded machine** (no concurrent jobs).

## Bin catalog

| bin | what |
|---|---|
| `behavior` | **The APP/behavior suite.** Run any bot across the garbage scenarios. Start here. |
| `metric` | Fast single-config metric (one number out) for quick iteration loops. |
| `bench-marathon` | Marathon scoring speed for the greedy baseline vs the multi-ply beam (depths 1-3). |
| `cc2-baseline` | Cold Clear 2's APP via the **TBP referee** (needs `CC2_BIN`). Its `VERSUS=1` mode is **NOT a fair fight**: TBP has no garbage message, so every garbage dump forces a `stop`+`start` re-sync that cripples CC2 (the bin prints the same warning). Use it for infrastructure checks, never for publishable win-rates; the fair comparison is `cc2-native`. |
| `cc2-native` | CC2's **ported** evaluator head-to-head in our fair native arena. |

## Key env knobs (the `behavior` bin)

| var | default | meaning |
|---|---|---|
| `BOT` | `dt20` | `dt20` \| `cc2` \| `cc2custom` (uses `CC2_PARAMS`) \| `bf` \| `bfcustom` (best-first + `CC2_PARAMS`) \| `bflin` \| `lincustom` (`BOARD_PARAMS`+`REWARD_PARAMS`) |
| `SEEDS` | 24 | number of seeds (cut to 2-3 when benchmarking best-first; it's slow) |
| `BEAM_DEPTH` / `BEAM_WIDTH` | 2 / 16 | beam search params; `BEAM_DEPTH` also caps best-first depth |
| `NODE_BUDGET` | 4000 | best-first total node-expansion budget per decision |
| `CC2_PARAMS` | (none) | 11 comma-separated CC2 board-weight floats (for `cc2custom`/`bfcustom`) |

## Current SOTA snapshot (2026-06)

- **Best-first (`BestFirstPlanner`) is the strongest search**, dominating the beam on the tuned
  attack eval with the gap **growing in garbage** (faucet1/2 best-first ≫ beam). But the
  **eval ≫ the search**: swapping linear→CC2 moves APP far more than any search change. The
  weight ablation shows `holes` and `row_transitions` carry the eval; the `height_upper_*`
  penalties are survival insurance that slightly *cost* APP.
- **Clean APP caps ~0.68**: the *eval's* ceiling (both searches reach it). Going past it needs
  a better policy (RL / value-net), not more search.
- **Latency**: `SearchState` holds a `Copy` `BitBoard` (bit-AND collision, alloc-free forks);
  best-first runs at ~25 ms/piece clean, at **bit-identical** APP versus the dense-board
  search it replaced (which ran ~115 ms/piece). The remaining gap to a 10× is an
  architecture/hardware call, not tuning.

## Reproduce the headline comparison

```sh
cargo build --release -p tetr-research --bin behavior
P="-0.003447473,-1.5,-0.2,-0.36203036,-1.5,-5.0,0.3472633,0.1,1.5,4.4650807,4.0"  # == Cc2Weights::attack_tuned()'s board params
# beam vs best-first on the same tuned attack eval, same seeds:
BOT=cc2custom CC2_PARAMS="$P" BEAM_DEPTH=6 BEAM_WIDTH=16 SEEDS=3 ./target/release/behavior
BOT=bfcustom  CC2_PARAMS="$P" BEAM_DEPTH=6 NODE_BUDGET=400 SEEDS=3 ./target/release/behavior
```

> **Long jobs:** best-first sweeps are slow; bound every run (`SEEDS` small, monitor wall-clock).
> There is no GNU `timeout` on macOS; self-bound and watch.
