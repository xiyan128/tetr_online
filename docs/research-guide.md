# tetr-research user guide

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
run ID, settings, numbers, verdict). Records are conclusions, not pending reruns.
Note it in the header if a later change breaks trajectory reproduction.

## Building a bot: `BotSpec`

The only construction path is `bots::BotSpec`: search × eval × sight, built
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
- `Greedy` + a custom eval **panics by design**; compose `beam`/`best_first`
  instead.
- A new evaluator gets a new `EvalSpec` arm, never a bypass.

## The harnesses (library)

| call | suite | measures |
|---|---|---|
| `marathon::evaluate[_capped]` | solo Marathon | score/sec, APP |
| `downstack::evaluate_downstack` | seeded cheese | censored pieces (optimization), clear rate, cleared-only pieces and attack (context) |
| `versus::evaluate_versus[_format]` | head-to-head, engine garbage rules | wins, deaths, net attack |
| `behavior::evaluate_scenario` | scripted pressure scenarios | APP, DS/P, survival, clear histogram |
| `sprt::sprt_race` | sequential survival test | H1 / H0 / inconclusive + LLR |

`VersusFormat { max_plies, rain_period }`: rain queues one cancellable line to
both sides every N plies. It is the decisiveness dial (mirror matches almost
never kill without it; rain 8 ≈ 98% decisive). All `evaluate_*` paths and the
SPRT run rayon-parallel (~6×), bit-identical to sequential by gate.

`versus_legacy::` is quarantined on purpose: the pre-engine garbage scheduler
kept ONLY for the TBP referee and the behavior faucet. Its rules diverge from
the engine's; never use it for a new experiment.

## The bins (experiments)

Run as `ENVVARS cargo run --release -p tetr-research --bin <name>`. Every bin
documents its env knobs and run records in its doc header. Read the header
before running.

**Daily drivers**

- **`metric`**: fast headline metrics for iteration loops. Default: capped
  marathon score/sec and APP. `DOWNSTACK=1` → censored pieces plus clear rate;
  `VERSUS=1` → win rate vs the greedy baseline. Knobs: `BENCH_SEEDS`,
  `BEAM_DEPTH`, `BEAM_WIDTH`.
- **`behavior`**: the APP/DS-P suite across the standard scenarios.
  `BOT=dt20|cc2|cc2custom|lincustom|bf|bfcustom|bflin` picks the spec;
  custom weights via `BOARD_PARAMS`/`REWARD_PARAMS`/`CC2_PARAMS` (CSV).
- **`bench-marathon`**: the full greedy-vs-beam sweep (depths 1-3).

**Versus science**

- **`garbage_ab`**: the awareness A/B. A spec vs its `.blind()` twin,
  arm-swapped, deaths split from cap tiebreaks. `BOT=beam|bf`,
  `WEIGHTS=attack`, `RAIN_PERIOD`, `SEEDS`.
- **`versus_climb`**: the (1+1)-ES weight climb with the three-stage gate
  chain — a fresh-block screen (`ACCEPT_MARGIN`, calibrate to ~2σ ≈ 150 at
  48 matches), a per-accept SPRT confirmation race (`CONFIRM_MATCHES`, 0
  disables; `CONFIRM_ALPHA` 0.02), and every `ANCHOR_EVERY` confirmed
  accepts an anchor race against the last *verified* point that re-anchors
  on H1 and ROLLS BACK on H0 — so confirmation-alpha accumulation buys noise
  for at most one anchor window, never the campaign. Runs under a `CAMPAIGN`,
  checkpoints every iteration, and `RESUME=<run-dir>` continues the walk
  bit-identically. Read the run records in its header before climbing; each
  documents a failure mode (seed overfit, noise acceptance, …) the current
  design retires.
- **`versus_sprt`**: the standalone racer, a ship-grade verdict on one
  candidate vs the incumbent. The unit of evidence is the chair-swapped seed
  PAIR (pair-level GSPRT — per-game Bernoulli walks void their α under
  within-pair correlation; the `sprt` module header carries the simulation
  receipts). `P1` (effect size, default 0.55), `ALPHA`, `BLOCK_SEEDS` (24),
  `RAIN_PERIOD` (8). ~5 min to resolve a true 0.5/0.55 at default settings;
  an in-budget inconclusive means the effect is small. That *is* the answer.
- **`promote`**: the promotion panel — the only gate from "my climb accepted
  it" to "it is the better bot". Candidate vs {greedy, origin, incumbent} ×
  rain {0, 8}, one pair-GSPRT per cell on fresh campaign seeds: greedy and
  origin cells demand H1, incumbent cells demand non-regression, H0 or
  starved evidence anywhere rejects. `CAND_PARAMS` / `INCUMBENT_PARAMS`
  (CSV), `CELL_MATCHES`. `FINAL_VALIDATION=1` spends the never-iterated
  FINAL region — exactly once per external claim.

**External baseline**

- **`cc2-baseline`**: the real Cold Clear 2 binary as a TBP subprocess,
  refereed on our seeded bag and attack table. Needs `CC2_BIN=/path/to/cc2`;
  `SEEDS`, `PIECES`, `THINK_MS`. Uses legacy garbage rules by design; its
  win rates are NOT comparable with `play_versus` numbers.
- **`cc2-native`**: CC2's *ported evaluator* vs ours on our engine with real
  mutual garbage. The fair comparison, and the baseline to climb past. Its
  downstack output reports censored pieces and clear rate for both evaluators.

## Run manifests

Every owned experiment creates `runs/<run-id>/`, where the run ID is
`<UTC timestamp>-<bin>-<pid>`. The directory contains:

- `spec.json`: schema version, run ID, bin, UTC creation time, git commit and
  dirty state, host metadata, every resolved env value with its raw value and
  source, and the bin's bot/seed/format specification.
- `outcomes.jsonl`: one JSON object per game or seed result.
- `summary.json`: UTC finish time, exit reason, and headline aggregate fields.
- `checkpoint.json`: optional atomically replaced resume state for experiments
  that checkpoint.

`runs/` is ignored by git. A doc-header RUN RECORD cites its run ID so the
durable conclusion can be traced to the machine-readable manifest.

## Seed regions and campaigns

```text
TRAIN       0        train / screening / quick A/Bs
VALIDATION  4096     held-out verdicts (pre-campaign experiments)
SPRT        16384    the standalone racer
ROTATION    1<<20    climb screen blocks (pre-campaign trajectories)
CONFIRM     1<<50    climb confirmation races (pre-campaign trajectories)
campaigns   1<<51    one private 2^32 slab per campaign name
FINAL       1<<63    never iterated — one verdict per external claim
```

Claim a new constant in `seeds::regions` for a new experiment; never invent
an offset inline. `seed_set(n)` = train; `seed_set_from(region, n)` = anywhere.

Static regions keep one run honest; they cannot keep a researcher honest
across runs — every inspect-then-adjust cycle leaks bits into a fixed
validation set. So optimization work runs under a **campaign**:
`Campaign::derive(name)` maps the name to a private slab with its own
validation / anchor / promotion / rotation / confirmation sub-regions
(bounds-checked, loud on exhaustion; the slot lands in every run manifest).
Reuse a name to *continue* that campaign; pick a fresh name per goal so no
verdict is ever quoted on seeds an earlier decision saw. `CAMPAIGN=scratch`
(the default) is the shared sandbox and promises no cross-run freshness.

**FINAL** is the one region nothing reads during iteration. `promote`
unlocks it behind `FINAL_VALIDATION=1` for the last verdict before an
external claim; spending it on anything that feeds back into tuning is the
one unrecoverable mistake this map cannot prevent.

## Adding an experiment (checklist)

1. Compose arms as `BotSpec`s; pick or claim a seed region.
2. Thin bin: env via strict `cli` helpers, library calls, one
   machine-readable `println!` per headline number, context on stderr.
3. Bound it: `TIME_BUDGET_SECS` with an honest partial verdict.
4. Create a `RunLedger` after all env reads, append one outcome per game, and
   write the terminal summary. Cite the run ID in each doc-header RUN RECORD.
5. If it judges survival, race it (`sprt_race`). Don't eyeball block means;
   they pass noise at every size we've measured (σ ≈ ±90 at 48 matches).

## Reading results / gotchas

- Environment defaults apply only when a variable is unset. A set value that
  cannot be parsed, has the wrong CSV length, or is not an allowed choice prints
  `config error: ...` and exits 2; it never silently falls back.
- The downstack optimization target is `mean_pieces_censored`: failures count as
  `max_pieces`. Compare it only between runs with the same recorded cap, and read
  clear rate beside it. Cleared-only mean pieces remains descriptive context.
- Sequential verdicts count the chair-swapped pair as ONE observation. The
  report carries the legacy per-game LLR as a cross-check and a within-pair
  correlation estimate — positive correlation means the per-game model would
  have overclaimed (measured: a 60%-coupled null false-accepts at 13.7%
  under the per-game walk vs 4.5% under the pair test, both at nominal 5%).
- **Win rate without deaths is a cap-game artifact.** Check the deaths split
  (`garbage_ab` prints it; `VersusOutcome.a_topped/b_topped` carry it).
- **Mirror matches are bland** (≤6% decisive); asymmetric-style matches are
  ~59% death-decisive without rain. Choose the format for the question.
- **Blind beats aware today** (the mispricing record): no-garbage-world
  weights overprice risen rows. Any aware-bot work must re-price first.
- A wall-clock budget couples the *stopping point* (never any match result)
  to machine speed; crossed SPRT bounds are machine-independent.
- Timing readouts (`ms/piece`, elapsed) are the only output lines that vary
  between runs; everything else is byte-reproducible.
- Wall-clock figures in run records written before 2026-06-10 predate the
  release-profile retune (`opt-level z → 3`, ~1.9× native throughput); match
  results are unaffected (FP stays IEEE at every opt level, golden-gated),
  but cross-era timing comparisons need re-baselining.
