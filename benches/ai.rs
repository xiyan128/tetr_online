//! AI benchmarks.
//!
//! Covers the placement pipeline bottom-up: reachability `movegen` (the hot
//! path), the board `evaluate`-or, the greedy `plan` that ties them together, and
//! an end-to-end "bot plays N pieces" throughput driver for tuning weights and
//! search depth. Run with `cargo bench --bench ai`.

mod common;

use std::hint::black_box;

use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};

use common::{first_locked, first_placement, play_pieces, search_state, spawner, Scenario};
use tetr_online::ai::{
    movegen, Cc2Evaluator, EvalContext, Evaluator, GreedyPlanner, LinearEvaluator, Planner,
    SearchBudget,
};
use tetr_online::engine::classify_t_spin;

/// Reachable-placement enumeration — the search's inner loop. Throughput is the
/// number of placements found, so criterion reports placements/sec (and the count
/// itself shows how each board shape widens or narrows the frontier).
fn bench_movegen(c: &mut Criterion) {
    let mut group = c.benchmark_group("ai/movegen");
    for scenario in Scenario::ALL {
        let state = search_state(scenario);
        let spawn = spawner();
        let queue_front = state.queue.first().copied();

        let no_hold = movegen::generate(&state.board, &state.active).len() as u64;
        group.throughput(Throughput::Elements(no_hold.max(1)));
        group.bench_function(BenchmarkId::new("generate", scenario.name()), |b| {
            b.iter(|| black_box(movegen::generate(black_box(&state.board), black_box(&state.active))));
        });

        let with_hold =
            movegen::generate_with_hold(&state.board, &state.active, state.hold, queue_front, &spawn)
                .len() as u64;
        group.throughput(Throughput::Elements(with_hold.max(1)));
        group.bench_function(BenchmarkId::new("generate_with_hold", scenario.name()), |b| {
            b.iter(|| {
                black_box(movegen::generate_with_hold(
                    black_box(&state.board),
                    black_box(&state.active),
                    state.hold,
                    queue_front,
                    &spawn,
                ))
            });
        });
    }
    group.finish();
}

/// The board evaluator (`Value` feature extraction + `Reward` classification) on a
/// realistic locked-placement fixture, across scenarios.
fn bench_evaluate(c: &mut Criterion) {
    // Both shipped evaluators: the linear DT-20 vector vs the ported Cold Clear 2 eval.
    // The CC2 eval is the per-node cost the best-first attack bot pays ~35×/expansion,
    // so its absolute µs sets the search's compute/quality frontier.
    let linear = LinearEvaluator::default();
    let cc2 = Cc2Evaluator::default();
    let evals: [(&str, &dyn Evaluator); 2] = [("linear", &linear), ("cc2", &cc2)];
    let mut group = c.benchmark_group("ai/evaluate");
    for scenario in Scenario::ALL {
        let (lock, board, t_spin) = first_locked(scenario);
        for (name, eval) in &evals {
            group.bench_function(BenchmarkId::new(*name, scenario.name()), |b| {
                b.iter(|| {
                    black_box(eval.evaluate(
                        black_box(&lock),
                        black_box(&board),
                        black_box(t_spin),
                        black_box(EvalContext::default()),
                    ))
                });
            });
        }
    }
    group.finish();
}

/// The per-candidate **board machinery** the search pays for every child: clone the
/// state, `commit_placement` (place the piece + line-clear + queue/hold advance), and
/// the pre-lock T-spin classification. With the evaluator benched cheap (~0.4µs), these
/// dominate best-first's per-piece cost — the target of a future bitboard `SearchState`
/// (Copy clone, free `column_bits`, bit-op commit). Reported per scenario.
fn bench_transition(c: &mut Criterion) {
    let mut group = c.benchmark_group("ai/transition");
    for scenario in Scenario::ALL {
        let state = search_state(scenario);
        let placement = first_placement(&state);

        group.bench_function(BenchmarkId::new("clone", scenario.name()), |b| {
            b.iter(|| black_box(black_box(&state).clone()));
        });
        group.bench_function(BenchmarkId::new("commit", scenario.name()), |b| {
            b.iter_batched(
                || state.clone(),
                |mut s| black_box(s.commit_placement(black_box(&placement))),
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

/// The full greedy pick: movegen + per-candidate evaluate + argmax. This is the
/// per-piece cost the controller pays each time it replans.
fn bench_plan(c: &mut Criterion) {
    let eval = LinearEvaluator::default();
    let budget = SearchBudget::greedy();
    let mut group = c.benchmark_group("ai/plan");
    for scenario in Scenario::ALL {
        let state = search_state(scenario);
        group.bench_function(BenchmarkId::from_parameter(scenario.name()), |b| {
            let mut planner = GreedyPlanner::new();
            b.iter(|| black_box(planner.plan(black_box(&state), &eval, budget)));
        });
    }
    group.finish();
}

/// End-to-end: a flawless seeded bot plays `target` pieces against a fresh engine.
/// Throughput is pieces, so criterion reports pieces/sec — the headline number for
/// "did my evaluator/search change make the bot faster or slower overall".
fn bench_game_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ai/game_throughput");
    // Each sample plays many pieces, so a smaller sample count keeps the wall-clock
    // reasonable while staying well above criterion's statistical floor.
    group.sample_size(20);
    for target in [25usize, 50] {
        group.throughput(Throughput::Elements(target as u64));
        group.bench_function(BenchmarkId::from_parameter(target), |b| {
            b.iter(|| black_box(play_pieces(black_box(target))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_movegen,
    bench_evaluate,
    bench_transition,
    bench_plan,
    bench_game_throughput
);
criterion_main!(benches);
