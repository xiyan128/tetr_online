# tetr-research user guide

The headless experiment platform behind the versus bot. Bevy-free, depends
only on `tetr-core`, always run `--release` (a debug match is ~20× slower).
This guide is task-oriented; the *rules* live in the crate docs
(`crates/tetr-research/src/lib.rs`) and bind every experiment.

## The five rules (short form)

1. **Determinism.** A game is a pure function of `(BotSpec, seed)`. Every
   reported number must reproduce from `(commit, eval, bots…)` — all names.
2. **Seed regions.** Seeds come from disjoint index regions
   (`seeds::regions`): train selects, validation checks, confirmation proves.
   Never quote a number on seeds that influenced a decision that produced it.
3. **Self-bounding.** Long commands honour their wall-clock budget
   (`--budget-secs`) and exit with an honest partial verdict. Never start an
   unbounded run.
4. **Arm-swap + CRN.** Paired comparisons play each seed from both chairs on
   common random numbers.
5. **Death decides.** The capped-game net-attack tiebreak is structurally
   anti-defensive; survival verdicts come from death-decisive matches (the
   SPRT), never bare capped win rates.

Results worth keeping go into the command's doc header as a **RUN RECORD** (date,
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
let tp    = BotSpec::tp_beam(128, 9);       // transposition-pruned beam
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
| `sprt::sprt_race` | sequential survival test | H1 / H0 / inconclusive + LLR |

`VersusFormat { max_plies, rain_period }`: rain queues one cancellable line to
both sides every N plies. It is the decisiveness dial (mirror matches almost
never kill without it; rain 8 ≈ 98% decisive). All `evaluate_*` paths and the
SPRT run rayon-parallel (~6×), bit-identical to sequential by gate.

`versus_legacy::` is quarantined on purpose: the pre-engine garbage scheduler
kept ONLY for the TBP referee and the behavior faucet. Its rules diverge from
the engine's; never use it for a new experiment.

## The CLI and the registries

Two registries, read as code (there is no `list`/`show` — the files ARE the
catalogs): **bots** (`src/bots.rs` — named `BotSpec` instances: who plays)
and **evals** (`src/registry.rs` — named, typed measurement specs). The
binary pairs them at the prompt:

```text
cargo run --release -p tetr-research -- run downstack dt20
cargo run --release -p tetr-research -- run versus cc2-default dt20
cargo run --release -p tetr-research -- run race v3-candidate attack-tuned
duckdb -init scripts/research.sql      # analyze every run ever recorded
```

A recorded result reproduces from `(commit, eval, bots…)` — all names, all
stamped into the run receipt. Want different parameters or a new candidate?
Register a new name — a climbed candidate is ONE bot registration, after
which it is raceable and benchmarkable at the prompt. Never
mutate a name with recorded runs; dirty-tree runs are stamped in the
receipt. The only
flags are machine-local: `--budget-secs`, `--cc2-bin`, `--runs-root`, and
`--allow-dirty` — runs REFUSE a dirty tree (or no git checkout) by default,
because such runs are not re-runnable from `(commit, eval, bots…)`; the
bypass records an exploratory run, stamped `git.dirty` in its receipt.
Tracking is not a participant: the runner writes the receipt and installs
the event sink before dispatch; commands never see either.

Optimizers wrap the same primitives and run by name like everything else:
**`app-climb`** is a (1+1)-ES over a subject bot's full Cc2 weight surface
(board + reward params, 26 dims) on **censored APP** — `total_attack /
max_pieces` per game, so a top-out dilutes by the pieces it forfeited and
survival is priced into the objective. Accepts pass a paired
common-random-numbers t-gate; screening seeds rotate through the campaign's
private slab; every run ends by self-validating origin-vs-final weights on
the campaign's held-out region (the `app.origin_val` / `app.best_val` fields
of its JSON line). The walk is a pure function of `(commit, spec, subject)` —
budgets only truncate it, so re-running with a larger `--budget-secs`
extends the SAME walk. Promotion stays manual: register the printed params
as a new bot, then judge it on `marathon-holdout` (held-out VALIDATION
seeds; read it once per candidate, not in a loop) and a `race`.

**The stdout contract**: every run prints exactly ONE self-describing JSON
line — `{"run": <dir>, "eval": …, "bots": […], …headline metrics}` — and
nothing else (humans read stderr; bars are stderr and TTY-only). Pipe it to
`jq`, or follow `run` to the receipt and game stream; analysis beyond the
headline belongs to duckdb.

**Daily drivers**

- **`marathon` / `downstack` / `versus`**: fast headline metrics — score/sec
  + APP, censored cheese pieces + clear rate, win/death/attack head-to-head
  (arm-swapped; deaths first-class). `run marathon dt20` and
  `run downstack dt20` are the /autoresearch loops (parse the JSON line's
  `score_per_second` / `pieces_censored` fields). Awareness A/Bs
  are versus with a blinded twin: `run versus cc2-default cc2-default-blind`
  (mirrors are bland without rain — the decisiveness dial).

**Versus science**

- **`race`**: the standalone racer — `run race <candidate> <incumbent>`, a
  ship-grade verdict. The unit of evidence is the chair-swapped seed PAIR
  (pair-level GSPRT — per-game Bernoulli walks void their α under
  within-pair correlation; the `sprt` module header carries the simulation
  receipts). ~5 min to resolve a true 0.5/0.55 at default settings; an
  in-budget inconclusive means the effect is small. That *is* the answer.


**External baseline**

- **`cc2-baseline-app` / `cc2-baseline-downstack`**: the real Cold Clear 2
  binary as a TBP subprocess, refereed on our seeded bag and attack table
  (`--cc2-bin /path/to/cc2`). Uses legacy garbage rules by design; its win
  rates are NOT comparable with `play_versus` numbers.
- **`run versus cc2-default dt20`** (and `run downstack cc2-default`):
  CC2's *ported evaluator* vs ours on our engine with real mutual garbage —
  the fair comparison, and the baseline to climb past.

## Receipts, events, and duckdb

Every run creates `runs/<UTC timestamp>-<eval>-<pid>/` holding two files,
split by role — **receipts are parameters, events are facts, metrics are
queries**:

- `spec.json` — the reproducibility receipt: schema version, run ID, git
  commit + dirty state, the eval name, its full typed spec, the bot names,
  and the runtime flags.
- `games.jsonl` — the facts: one row of raw outcomes per match, fully
  normalized. No timestamps, run ids, modes, bot names, or derived results
  in rows — the path, the receipt (`bots` + a per-game `swapped` bit), and
  queries carry those; the one ordinal `n` survives because order is
  semantic (LLR folds) and not recoverable in duckdb. Seeds are hex strings
  (u64s corrupt through f64-only readers). Rows are emitted after
  order-stable collection, so the file is BYTE-IDENTICAL across replays —
  `diff` is a replay witness, and the smoke asserts it.

The invariants behind this split are an ADR (`docs/adr-data-architecture.md`)
— read it before extending the data system. Analysis is duckdb, not the
platform: `duckdb -init scripts/research.sql`
gives `runs` / `games` / `games_wide` views (run ids from filenames,
bot names reconstructed from receipts), and a live run streams with
`tail -f … | jq`. Parquet is
an optional later compaction (`COPY … TO`), never the write format. The
platform never reads events back — they observe runs, they don't steer
them. `runs/` is ignored by git; doc-header RUN RECORDs cite run IDs.

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

1. Register the arms as named bots (`src/bots.rs`) if they don't exist.
2. A serde-serialized `Spec` with bot SLOTS + thin `run(spec, bots…, rt)`
   in a `commands/` module: library calls, one machine-readable `println!`
   per headline number, context on stderr — no tracking.
3. Register the eval in `src/registry.rs` (including a tiny `smoke-*`
   variant if the smoke gate should cover it) and wire the kind in
   `main.rs`'s dispatch (slot count + usage string).
4. Bound it: honour `rt.budget(...)` with an honest partial verdict.
5. If it judges survival, race it (`sprt_race`). Don't eyeball block means;
   they pass noise at every size we've measured (σ ≈ ±90 at 48 matches).

## Reading results / gotchas

- There is nothing to misspell: experiments run by registry name (unknown
  names exit 2 listing nothing silently), specs are typed Rust, and the only
  flags are machine-local. If a knob would change results, it belongs in a
  NEW registry entry, not on the command line.
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
