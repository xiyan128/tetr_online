# tetr-research — user guide

The headless experiment platform behind the versus bot. Bevy-free, depends
only on `tetr-core`, always run `--release` (a debug match is ~20× slower).
This guide is task-oriented; the *rules* live in the crate docs
(`crates/tetr-research/src/lib.rs`) and bind every experiment.

## The five rules (short form)

1. **Determinism.** A game is a pure function of `(BotSpec, seed)`. Every
   reported number must reproduce from code + env.
2. **Seed regions.** Seeds come from disjoint index regions
   (`seeds::regions`): train selects, validation checks, confirmation proves.
   Never quote a number on seeds that influenced a decision that produced it.
3. **Self-bounding.** Long bins honour `TIME_BUDGET_SECS` and exit with an
   honest partial verdict. Never start an unbounded run.
4. **Arm-swap + CRN.** Paired comparisons play each seed from both chairs on
   common random numbers.
5. **Death decides.** The capped-game net-attack tiebreak is structurally
   anti-defensive; survival verdicts come from death-decisive matches (the
   SPRT), never bare capped win rates.

Results worth keeping go into the bin's doc header as a **RUN RECORD** (date,
settings, numbers, verdict). Records are conclusions, not pending reruns —
note it in the header if a later change breaks trajectory reproduction.

## Building a bot: `BotSpec`

The only construction path (`bots::BotSpec`) — search × eval × sight, built
at full strength (imperfection 0, no reaction delay), seeded:

```rust
use tetr_research::bots::BotSpec;
use tetr_core::ai::eval::Cc2Weights;

let aware = BotSpec::beam(16, 2).cc2(Cc2Weights::attack_tuned());
let blind = aware.blind();                  // same brain, queue hidden
let deep  = BotSpec::best_first(4000, 6);   // node-budgeted graph search
let base  = BotSpec::greedy();              // the shipped Tier-1 baseline

evaluate_versus_format(&aware.factory(), &blind.factory(), &seeds, format);
```

- `.factory()` is what every harness function takes.
- `Greedy` + a custom eval **panics by design** — compose `beam`/`best_first`.
- A new evaluator gets a new `EvalSpec` arm, never a bypass.

## The harnesses (library)

| call | suite | measures |
|---|---|---|
| `marathon::evaluate[_capped]` | solo Marathon | score/sec, APP |
| `downstack::evaluate_downstack` | seeded cheese | pieces-to-clear (digging), attack-while-digging |
| `versus::evaluate_versus[_format]` | head-to-head, engine garbage rules | wins, deaths, net attack |
| `behavior::evaluate_scenario` | scripted pressure scenarios | APP, DS/P, survival, clear histogram |
| `sprt::sprt_race` | sequential survival test | H1 / H0 / inconclusive + LLR |

`VersusFormat { max_plies, rain_period }`: rain queues one cancellable line to
both sides every N plies — the decisiveness dial (mirror matches almost never
kill without it; rain 8 ≈ 98% decisive). All `evaluate_*` paths and the SPRT
run rayon-parallel (~6×), bit-identical to sequential by gate.

`versus_legacy::` is quarantined on purpose: the pre-engine garbage scheduler
kept ONLY for the TBP referee and the behavior faucet. Its rules diverge from
the engine's; never use it for a new experiment.

## The bins (experiments)

Run as `ENVVARS cargo run --release -p tetr-research --bin <name>`. Every bin
documents its env knobs and run records in its doc header — read the header
before running.

**Daily drivers**

- **`metric`** — one number, fast, for iteration loops. Default: capped
  marathon score/sec. `DOWNSTACK=1` → pieces-to-clear; `VERSUS=1` → win rate
  vs the greedy baseline. Knobs: `BENCH_SEEDS`, `BEAM_DEPTH`, `BEAM_WIDTH`.
- **`behavior`** — the APP/DS-P suite across the standard scenarios.
  `BOT=dt20|cc2|cc2custom|lincustom|bf|bfcustom|bflin` picks the spec;
  custom weights via `BOARD_PARAMS`/`REWARD_PARAMS`/`CC2_PARAMS` (CSV).
- **`bench-marathon`** — the full greedy-vs-beam sweep (depths 1–3).

**Versus science**

- **`garbage_ab`** — awareness A/B: a spec vs its `.blind()` twin,
  arm-swapped, deaths split from cap tiebreaks. `BOT=beam|bf`,
  `WEIGHTS=attack`, `RAIN_PERIOD`, `SEEDS`.
- **`versus_climb`** — the (1+1)-ES weight climb with the two-stage accept:
  fresh-block screen (`ACCEPT_MARGIN`, calibrate to ~2σ ≈ 150 at 48 matches)
  then an SPRT confirmation race (`CONFIRM_MATCHES`, 0 disables). Read the
  four run records in its header before climbing — each documents a failure
  mode (seed overfit, noise acceptance, …) the current design retires.
- **`versus_sprt`** — the standalone racer: ship-grade verdict on one
  candidate vs the incumbent. `P1` (effect size, default 0.55), `BLOCK_SEEDS`
  (24), `RAIN_PERIOD` (8). ~5 min to resolve a true 0.5/0.55 at default
  settings; an in-budget inconclusive means the effect is small — that *is*
  the answer.

**External baseline**

- **`cc2-baseline`** — the real Cold Clear 2 binary as a TBP subprocess,
  refereed on our seeded bag and attack table. Needs `CC2_BIN=/path/to/cc2`;
  `SEEDS`, `PIECES`, `THINK_MS`. Uses legacy garbage rules by design — its
  win rates are NOT comparable with `play_versus` numbers.
- **`cc2-native`** — CC2's *ported evaluator* vs ours on our engine with real
  mutual garbage — the fair comparison, and the baseline to climb past.

## Seed regions

```text
TRAIN       0        train / screening / quick A/Bs
VALIDATION  4096     held-out verdicts after optimization
SPRT        16384    the standalone racer
ROTATION    1<<20    climb screen blocks (one block per iteration)
CONFIRM     1<<50    climb confirmation races (stride scales with the cap)
```

Claim a new constant in `seeds::regions` for a new experiment; never invent
an offset inline. `seed_set(n)` = train; `seed_set_from(region, n)` = anywhere.

## Adding an experiment (checklist)

1. Compose arms as `BotSpec`s; pick or claim a seed region.
2. Thin bin: env via `cli::{env_usize, env_f64}`, library calls, one
   machine-readable `println!` per headline number, context on stderr.
3. Bound it: `TIME_BUDGET_SECS` with an honest partial verdict.
4. Doc header: purpose, env table, and a RUN RECORD after each real run.
5. If it judges survival, race it (`sprt_race`) — don't eyeball block means;
   they pass noise at every size we've measured (σ ≈ ±90 at 48 matches).

## Reading results / gotchas

- **Win rate without deaths is a cap-game artifact.** Check the deaths split
  (`garbage_ab` prints it; `VersusOutcome.a_topped/b_topped` carry it).
- **Mirror matches are bland** (≤6% decisive); asymmetric-style matches are
  ~59% death-decisive without rain. Choose the format for the question.
- **Blind beats aware today** (the mispricing record): no-garbage-world
  weights overprice risen rows. Any aware-bot work must re-price first.
- A wall-clock budget couples the *stopping point* (never any match result)
  to machine speed; crossed SPRT bounds are machine-independent.
- Timing readouts (`ms/piece`, elapsed) are the only output lines that vary
  between runs — everything else is byte-reproducible.
- Wall-clock figures in run records written before 2026-06-10 predate the
  release-profile retune (`opt-level z → 3`, ~1.9× native throughput); match
  results are unaffected (FP stays IEEE at every opt level — golden-gated),
  but cross-era timing comparisons need re-baselining.
