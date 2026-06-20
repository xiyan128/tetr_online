# Benchmarks

Criterion micro- and macro-benchmarks for the engine and AI. The suite is
**dev-only** — `criterion` is a `[dev-dependencies]` entry, so it never compiles
into `cargo build` or the size-optimized wasm/release binary.

## Running

```sh
cargo bench                       # everything
cargo bench --bench engine        # one target
cargo bench --bench ai            # the other

# filter by benchmark id (substring match against the group/parameter path):
cargo bench -- ai/movegen
cargo bench -- engine/primitives/lock_and_clear

# quick smoke run (short warm-up + measurement, fewer samples):
cargo bench --bench ai -- ai/plan --warm-up-time 0.5 --measurement-time 1 --sample-size 10
```

With the `html_reports` feature (enabled in `Cargo.toml`), criterion writes a
full report — plots, distributions, and run-to-run comparison — to
`target/criterion/`. Open `target/criterion/report/index.html`. Re-running a
bench compares against the previous run automatically and prints any regression
or improvement.

## Layout

```
benches/
  common/mod.rs   shared fixtures + helpers (NOT a bench target)
  engine.rs       Engine::step, snapshot, lock_and_clear, classify_t_spin
  ai.rs           movegen, evaluate, best-first plan, full-game throughput
```

`common/` lives in a subdirectory and the package sets `autobenches = false`, so
cargo never mistakes it for a benchmark — each bench target declares
`mod common;` to pull it in. Targets are listed explicitly under `[[bench]]` in
`Cargo.toml`.

## Conventions

- **Public API only.** Benches are separate crates; they touch `tetr_online`'s
  public surface exclusively. No internal access.
- **Deterministic.** Fixed seeds (`common::ENGINE_SEED`, `common::AI_SEED`), no
  wall-clock, no `rand::rng()`. Variance between runs is measurement noise.
- **Scenario spread.** Most benches loop over `common::Scenario::ALL` (empty →
  light stack → holey stack → near-top-out) so each result shows how cost scales
  with the board, not just one happy-path number.
- **Throughput where it means something.** `movegen` reports placements/sec,
  `game_throughput` reports pieces/sec — set via `group.throughput(...)`.
- **No state bleed.** Operations that mutate (e.g. `lock_and_clear`, a full game)
  use `iter_batched` with untimed setup so each sample starts clean.

## Adding a benchmark

1. **New measurement on existing fixtures** — add a `bench_*` fn to `engine.rs`
   or `ai.rs`, loop over `Scenario::ALL`, register it in that file's
   `criterion_group!`.
2. **New board shape** — add a `Scenario` variant, paint it in `common::paint`,
   name it in `Scenario::name`. Every scenario-looping bench picks it up.
3. **New subsystem** — add `benches/<name>.rs` with `mod common;` and a
   `criterion_main!`, then add a matching `[[bench]] name = "<name>"` /
   `harness = false` to `Cargo.toml`.
