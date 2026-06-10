//! Versus weight climb: hill-climb CC2 board params against a fixed incumbent
//! under the rain format — the win-rate-climb instrument the roadmap calls for.
//!
//! Method: a (1+1)-ES with common random numbers. Each candidate plays the
//! incumbent on the SAME train seeds in BOTH orientations (arm swap), scored by
//! a dense paired objective: death verdict (±1000 — death is what decides
//! matches) plus the net-attack margin (the cap-game tiebreak). Proposals
//! jitter each param relative to its magnitude (plus a small absolute floor so
//! near-zero params can move); sigma adapts by an accept-rate rule. The climb
//! is fully reproducible from CLIMB_SEED, and self-bounded by TIME_BUDGET_SECS.
//!
//! A held-out validation (disjoint seeds) reports the honest verdict at the
//! end: wins by death, cap tiebreaks, and mean margin — climbed weights ship
//! only if validation clears, never on the train objective.
//!
//! RUN RECORD (2026-06-10, 1h, defaults, rain 0): train objective 0 → +376.4
//! over 127 iters / 7 accepts — and validation came back DEAD EVEN (deaths
//! 19-19, tiebreaks 11-14, margin −0.9): pure seed overfitting from reusing 24
//! CRN seeds across every candidate. The gate worked; nothing shipped. Two
//! durable facts from the run: the paired objective is exactly fair (the
//! identical-weights baseline evaluates to +0.0), and candidate-vs-incumbent
//! matches are ~59% death-decisive even WITHOUT rain (asymmetric styles kill;
//! only same-weights mirrors are bland) — the real format has live death
//! signal. Next run's regularization: more/rotating train seeds, an
//! acceptance bar above noise (re-evaluate the incumbent or SPRT per
//! candidate), and a periodic held-out check during the climb.
//!
//! RUN RECORD v2 (same day, ROTATE=1, ACCEPT_MARGIN=25, SIGMA=0.15): rotation
//! killed seed memorization but the bar was far below the objective's noise —
//! with ±1000 death spikes a 48-match block mean has σ ≈ ±90, so 25 let lucky
//! proposals through (20 accepts / 56 iters), the accept-rate rule then GREW
//! sigma, and the parameters random-walked (tslot3 4.9 → 17.3); validation:
//! deaths 14-16, tiebreaks 8-57, margin −14.8. Gate caught it again; nothing
//! shipped. CALIBRATION LESSON: ACCEPT_MARGIN must be ~2σ of the block mean
//! (~150-200 at these sizes), with SIGMA small (~0.08) so a rare false accept
//! cannot teleport the walk.
//!
//! RUN RECORD v3 (same day, ACCEPT_MARGIN=150, SIGMA=0.08): the calibrated
//! design behaved exactly as intended — 45 iters, ONE accept (+213.9 over a
//! +0.0 incumbent block), and the first POSITIVE held-out validation: deaths
//! 20-15, tiebreaks 32-28, margin +0.79. Not statistically significant
//! (20-15 of 35 decisive ⇒ p ≈ 0.25 one-sided), so NOT shipped — but the
//! candidate is a small, sane perturbation of attack_tuned worth a long SPRT:
//! [-0.0036628882, -1.5733862, -0.19578815, -0.34977585, -1.5387586,
//!  -5.149458, 0.3575636, 0.09665186, 1.550793, 4.4781384, 3.782923]
//! SPRT EPILOGUE (same day, `versus_sprt`): the v3 candidate was REJECTED —
//! H0 accepted in 270 s (266-269 of 544 decisive, LLR −2.99). The one accept
//! the calibrated climb produced in an hour was noise after all; the gate
//! chain (held-out validation → SPRT) worked end to end and nothing shipped.
//! Honest read: board-params-only jitter around attack_tuned has no cheap
//! survival gains at this match budget — further wins want a different lever
//! (re-priced garbage-aware weights, deeper search, or the NN round), not
//! more of this walk.
//!
//! Next: longer budgets (the bar means ~1 accept/hour — that is the honest
//! pace of real progress), or wire `versus_sprt` in as a second-stage
//! confirmer per accept (~5 min each; screen with cheap blocks first).
//!
//! Env: TIME_BUDGET_SECS (1800), SEEDS (24 train; the per-iter block size
//!      when rotating), VAL_SEEDS (32), ROTATE (1), ACCEPT_MARGIN (25),
//!      RAIN_PERIOD (8), MAX_PLIES (240), BEAM_DEPTH (2), BEAM_WIDTH (16),
//!      SIGMA (0.15), CLIMB_SEED (1).
//!
//! ROTATE=1 (the default, the v2 regularization): every iteration draws a
//! FRESH disjoint seed block and evaluates BOTH the incumbent and the proposal
//! on it (paired within the iteration, never reused across iterations — the
//! overfitting channel from the first run is structurally gone), accepting
//! only when the proposal beats the incumbent by ACCEPT_MARGIN on that fresh
//! block. ROTATE=0 reproduces the v1 fixed-seed climb.

use std::time::Instant;

use tetr_core::ai::Cc2Weights;
use tetr_core::player::PlayerController;
use tetr_research::cli::{env_usize, SplitMix64};
use tetr_research::{
    beam_cc2_weights_bot, evaluate_versus_format, seed_set, seed_set_from, VersusFormat,
};

/// One standard-normal draw (Box-Muller over the deterministic SplitMix64).
fn gauss(rng: &mut SplitMix64) -> f64 {
    let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The paired objective of `candidate` vs the incumbent over `seeds`, both
/// orientations: mean over matches of `1000·(they died − we died) + (our net
/// attack − theirs)`. Higher is better; the death term dominates by design
/// (death decides matches; margin orders the rest).
fn objective(
    params: &[f32; Cc2Weights::BOARD_PARAM_COUNT],
    width: usize,
    depth: u8,
    seeds: &[u64],
    format: VersusFormat,
) -> f64 {
    let weights = Cc2Weights::attack_tuned().with_board_params(params);
    let cand = move |s: u64| -> Box<dyn PlayerController> {
        beam_cc2_weights_bot(s, width, depth, weights)
    };
    let incumbent = move |s: u64| -> Box<dyn PlayerController> {
        beam_cc2_weights_bot(s, width, depth, Cc2Weights::attack_tuned())
    };

    let fwd = evaluate_versus_format(&cand, &incumbent, seeds, format);
    let rev = evaluate_versus_format(&incumbent, &cand, seeds, format);

    let mut total = 0.0f64;
    let mut n = 0u32;
    for o in &fwd.outcomes {
        let death = 1000.0 * (f64::from(o.b_topped) - f64::from(o.a_topped));
        total += death + f64::from(o.attack_a) - f64::from(o.attack_b);
        n += 1;
    }
    for o in &rev.outcomes {
        let death = 1000.0 * (f64::from(o.a_topped) - f64::from(o.b_topped));
        total += death + f64::from(o.attack_b) - f64::from(o.attack_a);
        n += 1;
    }
    total / f64::from(n.max(1))
}

fn main() {
    let budget_secs = env_usize("TIME_BUDGET_SECS", 1800) as u64;
    let train_seeds = seed_set(env_usize("SEEDS", 24));
    let val_seeds = seed_set_from(4096, env_usize("VAL_SEEDS", 32));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let format = VersusFormat {
        max_plies: env_usize("MAX_PLIES", 240) as u32,
        rain_period: env_usize("RAIN_PERIOD", 8) as u32,
    };
    let mut sigma = std::env::var("SIGMA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.15);
    let mut rng = SplitMix64::new(env_usize("CLIMB_SEED", 1) as u64);

    eprintln!(
        "Versus climb — beam(d{depth}, w{width}) vs attack_tuned | {} train seeds x2, rain {}, {} plies | budget {budget_secs}s",
        train_seeds.len(),
        format.rain_period,
        format.max_plies
    );

    let rotate = env_usize("ROTATE", 1) == 1;
    let accept_margin = env_usize("ACCEPT_MARGIN", 25) as f64;
    let block = train_seeds.len();

    let start = Instant::now();
    let mut best_params = Cc2Weights::attack_tuned().board_params();
    let mut best = objective(&best_params, width, depth, &train_seeds, format);
    eprintln!(
        "iter 0 | baseline objective {best:+.1} | {:.0}s/eval | rotate {rotate} margin {accept_margin}",
        start.elapsed().as_secs_f32()
    );

    let mut iter = 0u32;
    let mut accepts = 0u32;
    while start.elapsed().as_secs() < budget_secs {
        iter += 1;
        // Relative jitter + a small absolute floor (params span magnitudes
        // from ~0.003 to ~1.5; the floor lets near-zero params move and sign
        // flips stay possible).
        let mut proposal = best_params;
        for p in proposal.iter_mut() {
            let scale = (p.abs() as f64 + 0.02) * sigma;
            *p += (scale * gauss(&mut rng)) as f32;
        }

        let accepted;
        if rotate {
            // Fresh disjoint block this iteration; incumbent and proposal race
            // on it head-to-head (paired CRN within the iteration, no reuse
            // across iterations — seed overfitting is structurally impossible).
            let block_seeds = seed_set_from(8192 + (iter as usize) * block, block);
            let incumbent_score = objective(&best_params, width, depth, &block_seeds, format);
            let proposal_score = objective(&proposal, width, depth, &block_seeds, format);
            accepted = proposal_score > incumbent_score + accept_margin;
            if accepted {
                eprintln!(
                    "iter {iter} | ACCEPT {proposal_score:+.1} vs {incumbent_score:+.1} | sigma {sigma:.3} | params {:?}",
                    proposal
                );
            } else {
                eprintln!(
                    "iter {iter} | reject {proposal_score:+.1} vs {incumbent_score:+.1} | sigma {sigma:.3}"
                );
            }
        } else {
            let score = objective(&proposal, width, depth, &train_seeds, format);
            accepted = score > best;
            if accepted {
                best = score;
                eprintln!(
                    "iter {iter} | ACCEPT {score:+.1} | sigma {sigma:.3} | params {:?}",
                    proposal
                );
            } else {
                eprintln!("iter {iter} | reject {score:+.1} (best {best:+.1}) | sigma {sigma:.3}");
            }
        }

        if accepted {
            accepts += 1;
            best_params = proposal;
            // One-fifth-style adaptation: widen on success, narrow on failure.
            sigma = (sigma * 1.3).min(0.5);
        } else {
            sigma = (sigma * 0.95).max(0.02);
        }
    }
    eprintln!("climb done: {iter} iters, {accepts} accepts");
    println!(
        "best_params {}",
        best_params
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Held-out validation: the honest verdict on DISJOINT seeds.
    let val = |params: &[f32; Cc2Weights::BOARD_PARAM_COUNT]| -> (u32, u32, u32, u32, f64) {
        let weights = Cc2Weights::attack_tuned().with_board_params(params);
        let cand = move |s: u64| -> Box<dyn PlayerController> {
            beam_cc2_weights_bot(s, width, depth, weights)
        };
        let incumbent = move |s: u64| -> Box<dyn PlayerController> {
            beam_cc2_weights_bot(s, width, depth, Cc2Weights::attack_tuned())
        };
        let fwd = evaluate_versus_format(&cand, &incumbent, &val_seeds, format);
        let rev = evaluate_versus_format(&incumbent, &cand, &val_seeds, format);
        let (mut dw, mut dl, mut margin) = (0u32, 0u32, 0.0f64);
        let (mut cap_w, mut cap_l) = (0u32, 0u32);
        for o in &fwd.outcomes {
            if o.b_topped && !o.a_topped {
                dw += 1;
            } else if o.a_topped && !o.b_topped {
                dl += 1;
            } else if o.attack_a > o.attack_b {
                cap_w += 1;
            } else if o.attack_b > o.attack_a {
                cap_l += 1;
            }
            margin += f64::from(o.attack_a) - f64::from(o.attack_b);
        }
        for o in &rev.outcomes {
            if o.a_topped && !o.b_topped {
                dw += 1;
            } else if o.b_topped && !o.a_topped {
                dl += 1;
            } else if o.attack_b > o.attack_a {
                cap_w += 1;
            } else if o.attack_a > o.attack_b {
                cap_l += 1;
            }
            margin += f64::from(o.attack_b) - f64::from(o.attack_a);
        }
        (
            dw,
            dl,
            cap_w,
            cap_l,
            margin / (2.0 * val_seeds.len() as f64),
        )
    };
    let (dw, dl, cw, cl, margin) = val(&best_params);
    eprintln!(
        "VALIDATION (held-out, {} seeds x2): deaths won {dw} lost {dl} | cap tiebreaks won {cw} lost {cl} | mean margin {margin:+.2}",
        val_seeds.len()
    );
    println!("val_death_wins {dw}");
    println!("val_death_losses {dl}");
    println!("val_mean_margin {margin:+.3}");
}
