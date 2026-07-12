//! Forward-throughput microbench — the datagen bottleneck is the net leaf eval
//! (CC2 datagen 31k games/hr vs net datagen 123: a 250x gap that IS the
//! forward). This reports evals/s across batch sizes so we can see whether the
//! forward is compute-bound (scales flat with batch) or overhead-bound (scales
//! up with batch = per-call fixed cost dominates small groups).
//!
//! Run (single-threaded, on the real trained net):
//!   TETR_BENCH_NET=~/leapfrog-rounds/r1/net \
//!     cargo test --release -p tetr-nn --test forward_bench -- --ignored --nocapture

use std::time::Instant;
use tetr_nn::net::{Net, Scratch};
use tetr_nn::obs::{BOARD_LEN, FEATURE_LEN};

fn synthetic(n: usize) -> (Vec<[f32; BOARD_LEN]>, Vec<[f32; FEATURE_LEN]>) {
    // Deterministic pseudo-random boards/features (no rand dep).
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let boards: Vec<[f32; BOARD_LEN]> = (0..n)
        .map(|_| std::array::from_fn(|_| if next() & 1 == 0 { 1.0 } else { 0.0 }))
        .collect();
    let feats: Vec<[f32; FEATURE_LEN]> = (0..n)
        .map(|_| std::array::from_fn(|_| (next() % 8) as f32))
        .collect();
    (boards, feats)
}

#[test]
#[ignore]
fn forward_evals_per_second() {
    let dir = std::env::var("TETR_BENCH_NET")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pyref").into());
    let dir = shellexpand(&dir);
    let net = Net::load(&dir).expect("bench net loads");
    let mut s = Scratch::default();

    // Warm caches / branch predictors.
    let (wb, wf) = synthetic(64);
    let witems: Vec<_> = wb.iter().zip(&wf).map(|(b, f)| (b, f)).collect();
    for _ in 0..20 {
        net.forward(&witems, &mut s);
    }

    eprintln!("\nnet {dir} — single-threaded forward throughput:");
    eprintln!("  {:>6}  {:>12}  {:>10}", "batch", "evals/s", "us/eval");
    for &n in &[1usize, 8, 34, 68, 128, 480, 1024] {
        let (boards, feats) = synthetic(n);
        let items: Vec<_> = boards.iter().zip(&feats).map(|(b, f)| (b, f)).collect();
        // Enough iterations that each batch runs ~0.3s.
        let mut iters = 0u64;
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 0.4 {
            net.forward(&items, &mut s);
            iters += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        let evals = iters as f64 * n as f64;
        eprintln!(
            "  {n:>6}  {:>12.0}  {:>10.2}",
            evals / secs,
            secs / evals * 1e6
        );
    }
    eprintln!(
        "\n  (datagen sibling groups are ~34-68 items; a beam GENERATION is ~8x that.\n   \
         If evals/s climbs steeply with batch, small per-call batches are the tax.)"
    );

    // Aggregate throughput at W concurrent threads (each its own Net+Scratch,
    // like datagen workers). If total evals/s does NOT scale ~W, the threads
    // are contending — the classic BLAS-thread-oversubscription pathology
    // (each sgemm spawns Accelerate threads; W workers => W*P threads on 12
    // cores). Run this test twice — with and without VECLIB_MAXIMUM_THREADS=1
    // in the env — to see if capping BLAS to one thread per worker fixes it.
    eprintln!(
        "\nconcurrent-worker aggregate throughput (batch 50, VECLIB_MAXIMUM_THREADS={}):",
        std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "unset".into())
    );
    eprintln!(
        "  {:>8}  {:>12}  {:>10}",
        "workers", "total ev/s", "per-worker"
    );
    for &w in &[1usize, 2, 4, 8, 10] {
        let total: f64 = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..w)
                .map(|_| {
                    scope.spawn(|| {
                        let net = Net::load(&dir).expect("net");
                        let mut s = Scratch::default();
                        let (boards, feats) = synthetic(50);
                        let items: Vec<_> =
                            boards.iter().zip(&feats).map(|(b, f)| (b, f)).collect();
                        let mut iters = 0u64;
                        let t0 = Instant::now();
                        while t0.elapsed().as_secs_f64() < 0.5 {
                            net.forward(&items, &mut s);
                            iters += 1;
                        }
                        iters as f64 * 50.0 / t0.elapsed().as_secs_f64()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        eprintln!("  {w:>8}  {total:>12.0}  {:>10.0}", total / w as f64);
    }
    eprintln!(
        "  (ideal: total scales ~W, per-worker flat. If per-worker COLLAPSES as\n   \
         W grows, workers are fighting over BLAS/AMX threads — cap fixes it.)"
    );
}

/// Minimal `~` expansion (no dep).
fn shellexpand(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}
