//! Forward-throughput microbench (T03 budget anchor). Not a correctness test —
//! measures batched-forward evals/s on the round0 net at a sweep of batch sizes,
//! to price the leaf-eval cost a net-guided search pays. Run with:
//!   cargo test --release -p tetr-nn --test throughput -- --ignored --nocapture

use std::time::Instant;

use tetr_nn::net::{Net, Scratch};
use tetr_nn::obs::{BOARD_LEN, BOARD_W, FEATURE_LEN};

fn plane(seed: u64) -> [f32; BOARD_LEN] {
    // A plausible stack: bottom 12 rows ~60% full with a moving hole column.
    let mut p = [0.0f32; BOARD_LEN];
    let mut s = seed | 1;
    for row in (BOARD_LEN / BOARD_W - 12)..(BOARD_LEN / BOARD_W) {
        let hole = (s % BOARD_W as u64) as usize;
        for x in 0..BOARD_W {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if x != hole && !(s >> 33).is_multiple_of(5) {
                p[row * BOARD_W + x] = 1.0;
            }
        }
    }
    p
}

fn feats(seed: u64) -> [f32; FEATURE_LEN] {
    let mut f = [0.0f32; FEATURE_LEN];
    for (i, v) in f.iter_mut().enumerate() {
        *v = ((seed.wrapping_add(i as u64) % 7) as f32) * 0.1;
    }
    f
}

#[test]
#[ignore]
fn forward_throughput_sweep() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/round0");
    let net = Net::load(dir).expect("round0 fixture loads");
    let mut s = Scratch::default();

    // One frozen opponent embedding, as serving does per decision.
    let opp_plane = plane(999);
    let opp = net.embed_boards(&[&opp_plane], &mut s).pop().unwrap();

    let max_n = 4096usize;
    let planes: Vec<[f32; BOARD_LEN]> = (0..max_n).map(|i| plane(i as u64)).collect();
    let features: Vec<[f32; FEATURE_LEN]> = (0..max_n).map(|i| feats(i as u64)).collect();

    println!("\nbatch,evals_per_s,us_per_batch,us_per_eval");
    for &n in &[1usize, 8, 16, 34, 64, 128, 256, 480, 1024, 2048, 4096] {
        let items: Vec<(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])> =
            (0..n).map(|i| (&planes[i], &features[i])).collect();
        // Warm up.
        for _ in 0..3 {
            std::hint::black_box(net.forward(&items, &opp, &mut s));
        }
        // Time ~0.4s of work.
        let mut iters = 0u64;
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 0.4 {
            std::hint::black_box(net.forward(&items, &opp, &mut s));
            iters += 1;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let evals = iters * n as u64;
        let eps = evals as f64 / elapsed;
        let us_batch = elapsed / iters as f64 * 1e6;
        let us_eval = us_batch / n as f64;
        println!("{n},{eps:.0},{us_batch:.1},{us_eval:.3}");
    }
}
