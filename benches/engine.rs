//! Engine-core benchmarks.
//!
//! Covers the per-frame driver (`Engine::step`), snapshotting (rebuilt every
//! frame for the renderer), and the pure rule primitives the AI reuses to
//! simulate placements. Run with `cargo bench --bench engine`.

mod common;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use common::{SIM_DT, Scenario, fresh_engine, search_state};
use tetr_online::engine::{InputFrame, classify_t_spin, lock_and_clear};

/// `Engine::step` for a representative set of single-frame inputs. Each iteration
/// steps a *fresh* engine (set up untimed) so state never accumulates across
/// samples. Throughput is 1 step, so criterion reports steps/sec.
fn bench_step(c: &mut Criterion) {
    let inputs: [(&str, InputFrame); 4] = [
        ("neutral", frame(|_| {})),
        ("move", frame(|f| f.left = true)),
        ("rotate", frame(|f| f.rotate_clockwise = true)),
        ("hard_drop", frame(|f| f.hard_drop = true)),
    ];

    let mut group = c.benchmark_group("engine/step");
    group.throughput(Throughput::Elements(1));
    for (name, input) in inputs {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_batched(
                fresh_engine,
                |mut engine| black_box(engine.step(input.clone())),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `Engine::snapshot` across board scenarios — cost scales with occupied cells,
/// and it runs once per rendered frame, so it is worth watching.
fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/snapshot");
    for scenario in Scenario::ALL {
        let engine = common::scenario_engine(scenario);
        group.bench_function(BenchmarkId::from_parameter(scenario.name()), |b| {
            b.iter(|| black_box(engine.snapshot()));
        });
    }
    group.finish();
}

/// The pure rule primitives the AI calls for every candidate placement:
/// `lock_and_clear` (mutates a board — cloned fresh per iteration) and
/// `classify_t_spin` (read-only). Benched across scenarios since both scale with
/// board occupancy.
fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine/primitives");
    for scenario in Scenario::ALL {
        let state = search_state(scenario);
        let placement = common::first_placement(&state);

        group.bench_function(BenchmarkId::new("lock_and_clear", scenario.name()), |b| {
            b.iter_batched(
                || state.board.to_array2d(),
                |mut board| black_box(lock_and_clear(&placement.piece, &mut board)),
                BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("classify_t_spin", scenario.name()), |b| {
            b.iter(|| {
                black_box(classify_t_spin(
                    black_box(&placement.piece),
                    black_box(&state.board),
                ))
            });
        });
    }
    group.finish();
}

/// Build an `InputFrame` carrying one sim step of `dt`, mutated by `f`.
fn frame(f: impl FnOnce(&mut InputFrame)) -> InputFrame {
    let mut frame = InputFrame {
        dt_seconds: SIM_DT,
        ..InputFrame::default()
    };
    f(&mut frame);
    frame
}

criterion_group!(benches, bench_step, bench_snapshot, bench_primitives);
criterion_main!(benches);
