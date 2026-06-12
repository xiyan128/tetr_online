# tetr-research agent-ops foundation: strict config, safe downstack metric, run ledger

Target: `crates/tetr-research`, the Bevy-free headless experiment platform that depends
only on `tetr-core`. Before implementation, read `crates/tetr-research/src/lib.rs` and
`docs/research-guide.md`.

## Context

An external audit verified three foundation problems. A parallel workstream covering
pair-aware SPRT statistics, climb checkpoint/resume, campaign seed regions, and the
promotion suite will build on this work. Avoid interface drift, especially in the pinned
ledger API below.

1. **The downstack optimization metric is gameable.**
   `src/downstack.rs:126-135` averages pieces-to-clear over games that cleared only and
   returns `0.0` when none cleared. `src/bin/metric.rs:29` exposes that mean as the single
   machine-parsed stdout metric while `clear_rate` goes to stderr. An optimizer can favor
   never-clearing or selective failure on hard seeds.
2. **Config fails open.**
   `src/cli.rs:13` silently substitutes defaults when set environment variables fail to
   parse. `BOT` handling and weight-list variables have similar lenient paths, so a run
   record can describe an experiment that did not run as configured.
3. **There are no machine-readable run records.**
   Results live in bin doc headers and stderr. There is no manifest containing the git
   state, resolved config, per-seed outcomes, or summary for automation and resume.

## House Rules

- Branch from current `master`; never commit directly to `master`. Open a GitHub PR when
  implementation is complete.
- Before opening the PR, make the full `scripts/gate` green. It covers fmt, clippy with
  denied `dbg!`/`todo!`, tests, and rustdoc with `-D warnings`.
- Rustdoc links to renamed or deleted items must remain valid. Escape literal citation
  brackets such as `\[N\]` in doc comments.
- This is Rust edition 2024: `gen` is a keyword and clippy enforces let-chains.
- The crate has no external consumers and keeps no compatibility surface. Replace or
  rename outright; do not add deprecated shims.
- Preserve determinism: a game is a pure function of `(bot spec, seed)`. Change only
  aggregation, parsing, and I/O, never gameplay. Existing parallel-equals-sequential
  tests must remain green.
- `serde` and `serde_json` are already dependencies. Add no other dependencies.
- Comments should state constraints the code cannot express, not narrate the task.
- Never commit `runs/` artifacts.

## Owned Files

- `crates/tetr-research/src/cli.rs`
- `crates/tetr-research/src/downstack.rs`
- `crates/tetr-research/src/ledger.rs` (new)
- `crates/tetr-research/src/lib.rs`
- `crates/tetr-research/src/bin/metric.rs`
- `crates/tetr-research/src/bin/cc2_native.rs`
- `crates/tetr-research/src/bin/garbage_ab.rs`
- `crates/tetr-research/src/bin/behavior.rs`
- `crates/tetr-research/src/bin/bench_marathon.rs`
- `crates/tetr-research/src/bin/cc2_baseline.rs`
- One-line `#[derive(serde::Serialize)]` additions to outcome structs in
  `downstack.rs`, `marathon.rs`, `versus.rs`, and `behavior.rs` as needed
- `.gitignore` to add `/runs/`
- Only relevant subsections of `docs/research-guide.md`

Do not modify parallel-workstream files:

- `src/sprt.rs`
- `src/seeds.rs`
- `src/versus_legacy.rs`
- `src/bin/versus_climb.rs`
- `src/bin/versus_sprt.rs`

If implementation appears to require a forbidden file, stop and record the need in the
PR description.

## Task A: Strict, Recorded Environment Config

Keep the signatures of `env_or`, `env_usize`, and `env_f64`, but make them strict:

- Unset means use the default.
- Set and valid means use the parsed value.
- Set and invalid means print
  `config error: KEY="<raw>" is not a valid <type>` to stderr and exit with code 2.
- Implement the decision as a pure, unit-testable function returning `Result`; public
  helpers should be thin exiting wrappers.

Add and migrate call sites to:

- `env_flag(key) -> bool`: presence check for `DOWNSTACK` in `bin/metric.rs` and
  `bin/cc2_baseline.rs`, ensuring mode flags are recorded.
- `env_choice(key, default, allowed: &[&str]) -> String`: reject set-but-unknown values
  with exit code 2 and list allowed values. Replace `BOT` reads in `bin/garbage_ab.rs`
  and `bin/behavior.rs`; derive the exact allowed values from their match arms.

Audit all direct `std::env::var` reads in owned bins, including `WEIGHTS`,
`BOARD_PARAMS`, `REWARD_PARAMS`, and `CC2_BIN`. Invalid floats and element counts must
exit 2 through the same pure core, and every read must be registered.

Maintain a process-global provenance registry, for example
`OnceLock<Mutex<BTreeMap<...>>>`, recording:

```json
{
  "KEY": {
    "raw": null,
    "value": "resolved value",
    "source": "env or default"
  }
}
```

Expose the registry as `cli::resolved_env() -> serde_json::Value`. Panic if a key is read
again with a different default. The two `MAX_PIECES` reads in `bin/metric.rs` are in
mutually exclusive early-return branches; verify that only one is registered per run.

Tests:

- [ ] Parse succeeds.
- [ ] Unset value uses the default.
- [ ] Parse failure names the key and raw value.
- [ ] Unknown choice is rejected.
- [ ] Registry contents are correct after a sequence of reads.

## Task B: Censored Downstack Metric

In `downstack.rs`:

- Extract aggregation from `evaluate_downstack` into
  `DownstackStats::from_outcomes(outcomes: Vec<DownstackOutcome>, max_pieces: u32)`.
  Leave the parallel gameplay loop unchanged.
- Add `max_pieces: u32` and `mean_pieces_censored: f32` to `DownstackStats`.
- For `mean_pieces_censored`, every cleared game contributes `pieces`; every failed game
  contributes `max_pieces`, whether failure came from the cap or top-out.
- Retain `mean_pieces_to_clear` and `mean_attack` as descriptive context.
- Document that the censored mean is the optimization-safe target and is meaningful only
  together with its `max_pieces`.

In `bin/metric.rs`, deliberately break the old parsed contract in downstack mode:

```text
downstack_pieces_censored <x.xx>
downstack_clear_rate <x.xx>
```

Delete `downstack_pieces_to_clear`. Update header docs so guard metrics are on stdout.

In `bin/cc2_native.rs`, emit both metrics for both arms:

```text
downstack_cc2_pieces_censored <x.xx>
downstack_cc2_clear_rate <x.xx>
downstack_dt20_pieces_censored <x.xx>
downstack_dt20_clear_rate <x.xx>
```

Run `cc2_native` at defaults in release mode and refresh its doc-header record with the
new numbers, clear rates, and the ledger run ID. Apply code changes to `cc2_baseline`, but
do not re-run its recorded numbers because it needs the external `CC2_BIN`; flag that as
a PR follow-up.

Synthetic aggregation tests:

- [ ] All failures produce `mean_pieces_censored == max_pieces` and `clear_rate == 0.0`.
- [ ] With cap 100, bot A clears at 20, 22, 24, and 30 pieces while bot B fails only the
  30-piece seed. Assert A's censored mean is `24.0`, B's is `41.5`, and A ranks better,
  despite B's misleading cleared-only mean of `22.0`.
- [ ] A real `BotSpec::greedy()` with `max_pieces = 1` and `rows = 9` cannot clear, so
  censored mean is `1.0` and clear rate is `0.0`.

## Task C: Run Ledger

Create `crates/tetr-research/src/ledger.rs`. The following API is a pinned contract; list
any deviation prominently in the PR description.

```rust
pub struct RunLedger { /* ... */ }

impl RunLedger {
    /// Create `<runs-root>/<YYYYMMDD-HHMMSS>-<bin>-<pid>/` and write spec.json.
    /// runs-root = `git rev-parse --show-toplevel`/runs (fallback: ./runs).
    pub fn create(bin: &str, extra_spec: serde_json::Value) -> std::io::Result<RunLedger>;

    /// Same, rooted at an explicit dir (tests use this with a tempdir).
    pub fn create_at(
        root: &std::path::Path,
        bin: &str,
        extra_spec: serde_json::Value,
    ) -> std::io::Result<RunLedger>;

    /// Append one JSON object as one line of outcomes.jsonl.
    pub fn append_outcome(
        &mut self,
        outcome: &impl serde::Serialize,
    ) -> std::io::Result<()>;

    /// Write summary.json (once, at exit).
    pub fn write_summary(&self, summary: serde_json::Value) -> std::io::Result<()>;

    /// Atomically write/overwrite checkpoint.json via a temp file and fs::rename.
    pub fn write_checkpoint(&self, state: serde_json::Value) -> std::io::Result<()>;

    /// Read a checkpoint back from a run directory.
    pub fn read_checkpoint(
        run_dir: &std::path::Path,
    ) -> std::io::Result<serde_json::Value>;

    pub fn dir(&self) -> &std::path::Path;
}
```

`spec.json` mandatory fields:

- `schema_version: 1`
- `run_id`
- `bin`
- `created_utc` in RFC3339 format
- `git: { commit: string|null, dirty: bool|null }`, using `git rev-parse HEAD` and
  `git status --porcelain`; record nulls rather than failing when git commands fail
- `host: { hostname, cores, os }`
- `env` from `cli::resolved_env()`; construct the ledger after all environment reads
- `extra` with caller-provided bot specs, seed counts or regions, and formats

`summary.json` must include `schema_version`, `finished_utc`, `exit_reason` such as
`"complete"` or `"time_budget"`, plus caller fields.

Wire the ledger into owned bins only: `metric`, `cc2_native`, `garbage_ab`, `behavior`,
`bench_marathon`, and `cc2_baseline`. Write the spec after env reads, append one outcome
per game or seed, and write a summary at exit with headline statistics. Preserve all
existing stdout/stderr bytes except the intentional Task B contract change.

Housekeeping:

- [ ] Add `/runs/` to `.gitignore`.
- [ ] Add `pub mod ledger;` to `lib.rs`.
- [ ] Add the ledger to the `lib.rs` layout table.
- [ ] Add a manifest convention and amend the run-record convention so doc-header
  records cite run IDs.

Ledger test using an explicit temporary directory:

- [ ] `create_at` succeeds.
- [ ] Three outcomes produce three JSONL lines.
- [ ] Writing the checkpoint twice leaves the second value.
- [ ] Summary and all expected files exist.
- [ ] `spec.json` parses and contains the populated environment map.

## Documentation

Update only the affected parts of `docs/research-guide.md`:

- Changed bin-table entries.
- A short Run manifests subsection describing layout and fields.
- Strict-config behavior under rules or gotchas.

Do not modify statistics, campaign-seeds, or promotion sections owned by the parallel
workstream.

## Acceptance Checklist

- [ ] `cargo test -p tetr-research --locked` is green.
- [ ] Full `scripts/gate` is green.
- [ ] `DOWNSTACK=1 BENCH_SEEDS=4 cargo run --release -p tetr-research --bin metric`
  prints `downstack_pieces_censored` and `downstack_clear_rate`, and creates a complete
  run directory. Include `ls` and `spec.json` receipts in the PR description.
- [ ] `BENCH_SEEDS=nope cargo run -p tetr-research --bin metric` exits 2 with
  `config error: BENCH_SEEDS="nope" ...`. Include the receipt.
- [ ] `BOT=tabu cargo run -p tetr-research --bin <garbage_ab bin name>` exits 2 and lists
  allowed values. Verify the registered bin name; some are dashed, such as
  `cc2-baseline`.
- [ ] `cc2_native.rs` doc header contains censored numbers, clear rates, and a run ID.
- [ ] The `cc2_baseline` re-record is flagged as a follow-up.
- [ ] Any deviation from the pinned ledger API is explicitly listed.
- [ ] No forbidden files changed.
- [ ] No `runs/` artifacts are committed.
- [ ] Commits are small and conventional, for example `feat(research): ...`.
