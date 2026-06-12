//! `app-climb`: hill-climb a Cc2 beam bot's full weight surface on censored APP.
//!
//! The optimizer the search teardown promised back "wrapping the primitives":
//! a (1+1)-ES over the subject's **board + reward** parameters
//! ([`Cc2Weights::board_params`] ++ [`Cc2Weights::reward_params`], 26 dims —
//! the reward side, including the engine-true `attack` term, was never
//! climbable before). Fitness is **censored APP**: `total_attack / max_pieces`
//! per game, so a top-out dilutes by exactly the pieces it forfeited — the
//! survival constraint is priced into the objective instead of bolted on.
//! (Empty-board APP is the recorded gameable metric for cross-bot comparisons;
//! a climb may still use it as its fitness because the verdict that matters is
//! the held-out delta of the SAME metric, plus the downstack/versus context
//! evals the candidate must face before promotion.)
//!
//! Mechanics: candidate = incumbent + sparse Gaussian perturbation (per-dim
//! scale frozen at the origin's magnitudes); accept on a paired per-seed
//! t-gate (`mean Δ > k·SE`, common random numbers); step size follows a
//! 1/5th-style rule (×1.3 on accept, ×0.92 on reject). Screening seeds rotate
//! every `block_iters` iterations through the campaign's rotation sub-slab, so
//! no fixed seed set is climbed into; the run ends with a self-validation of
//! origin vs final params on the campaign's held-out validation region — the
//! honest readout, printed in the stdout line as `app.origin_val` / `best_val`.
//!
//! The walk is a pure function of `(commit, eval spec, subject)`: the ES RNG
//! seeds from the campaign slot, game seeds come from the campaign slab, and
//! budgets only truncate the walk (a budget-cut run is a prefix of the
//! unbounded one). Promotion stays manual by design: register the printed
//! params as a new named bot, then race it.
//!
//! # RUN RECORD (2026-06-12, campaign `app-1p0`, subject attack-tuned-d4)
//!
//! The first recorded climb (600 s, run `20260612-075448-app-climb-80407`,
//! commit d67b402): 403 iters, 25 accepts, held-out **0.5971 → 0.6033**
//! (+0.006 ≈ noise) — a NULL result, and a diagnostic one. σ collapsed to the
//! then-0.01 floor within ~40 iters (see the `sigma_decay` field doc), and the
//! same session's single-lever probes (`bots.rs`, probe-*) independently found
//! every eval-side direction loses or ties at the d6w32 incumbent: the
//! attack-tuned weights are locally optimal for censored APP. The lever that
//! moved APP all session was SEARCH CLASS (depth → width → best-first →
//! transposition-pruned beam), not weights — spend climb budget there first.

use std::io;
use std::time::Instant;

use serde_json::{Value, json};
use tetr_core::ai::eval::Cc2Weights;

use crate::bots::{Bot, BotSpec, EvalSpec, SearchSpec};
use crate::commands::Runtime;
use crate::events;
use crate::marathon::{DEFAULT_MAX_FRAMES, MarathonOutcome, evaluate_capped};
use crate::rng::SplitMix64;
use crate::seeds::{Campaign, seed_set_from};

/// Default wall-clock budget (`--budget-secs` overrides): long enough for a few
/// hundred d6w32 iterations, short enough to never need interrupting.
const DEFAULT_BUDGET_SECS: u64 = 600;

const BOARD: usize = Cc2Weights::BOARD_PARAM_COUNT;
const DIMS: usize = BOARD + Cc2Weights::REWARD_PARAM_COUNT;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Campaign name — owns the rotation/validation seed slab AND seeds the ES
    /// RNG (via its derived slot), so the whole walk replays from the spec.
    pub campaign: &'static str,
    /// Games per fitness evaluation (CRN-paired across the two arms).
    pub seeds_per_block: usize,
    /// Iterations between screening-block rotations.
    pub block_iters: u32,
    /// Per-game piece cap — the censored-APP denominator (metric definition).
    pub max_pieces: u32,
    /// Iteration cap (the budget usually binds first).
    pub iters: u32,
    /// Initial relative step size.
    pub sigma0: f32,
    /// Per-reject step decay and the step floor. RUN RECORD (the first 600s
    /// d4 climb, run `20260612-075448-app-climb-80407`): decay 0.92 with
    /// floor 0.01 collapsed σ to the floor within ~40 iters under the ~6%
    /// accept rate the t-gate produces — a reject mostly means "no detectable
    /// improvement", not "step too big", so the 1/5th-rule assumption fails
    /// and the walk explored a ±1% ball (held-out Δ +0.006 ≈ noise). Keep the
    /// floor high enough that proposals stay meaningful.
    pub sigma_decay: f32,
    pub sigma_floor: f32,
    /// Accept gate: paired mean Δ must exceed `accept_k × SE`.
    pub accept_k: f64,
    /// Held-out games per arm for the end-of-run self-validation.
    pub val_seeds: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            campaign: "app-1p0",
            seeds_per_block: 8,
            block_iters: 8,
            max_pieces: 150,
            iters: 1000,
            sigma0: 0.15,
            sigma_decay: 0.97,
            sigma_floor: 0.05,
            accept_k: 1.0,
            val_seeds: 16,
        }
    }
}

/// The subject's climbable surface: its beam config + Cc2 weights.
fn subject_surface(subject: &Bot) -> io::Result<(usize, u8, Cc2Weights)> {
    match (subject.spec.search, subject.spec.eval) {
        (SearchSpec::Beam { width, depth }, EvalSpec::Cc2(w)) => Ok((width, depth, w)),
        _ => Err(io::Error::other(format!(
            "app-climb needs a beam + Cc2 subject; {} is {:?}",
            subject.name, subject.spec
        ))),
    }
}

fn to_vec(w: &Cc2Weights) -> [f32; DIMS] {
    let mut v = [0.0; DIMS];
    v[..BOARD].copy_from_slice(&w.board_params());
    v[BOARD..].copy_from_slice(&w.reward_params());
    v
}

/// Rebuild weights from a param vector, inheriting the subject's fixed fields
/// (`max_cell_covered_height`, `perfect_clear_override`, `softdrop`).
fn to_weights(origin: &Cc2Weights, v: &[f32; DIMS]) -> Cc2Weights {
    let board: [f32; BOARD] = v[..BOARD].try_into().expect("board slice");
    let reward: [f32; Cc2Weights::REWARD_PARAM_COUNT] =
        v[BOARD..].try_into().expect("reward slice");
    origin.with_board_params(&board).with_reward_params(&reward)
}

/// ~N(0,1): sum of 12 uniforms − 6 (deterministic, dependency-free; tails
/// clipped at ±6, irrelevant at our step sizes).
fn normal(rng: &mut SplitMix64) -> f32 {
    let mut s = 0.0f32;
    for _ in 0..12 {
        s += (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
    }
    s - 6.0
}

/// Sparse Gaussian move: each dim perturbs with probability ~3/DIMS (resampled
/// until at least one moves), step = `sigma × scale × N(0,1)`.
fn perturb(
    cur: &[f32; DIMS],
    scales: &[f32; DIMS],
    sigma: f32,
    rng: &mut SplitMix64,
) -> [f32; DIMS] {
    let mut next = *cur;
    loop {
        let mut moved = false;
        for i in 0..DIMS {
            if rng.next_u64() % (DIMS as u64) < 3 {
                next[i] = cur[i] + sigma * scales[i] * normal(rng);
                moved = true;
            }
        }
        if moved {
            return next;
        }
    }
}

/// One fitness evaluation: the subject's beam over `weights`, censored APP per
/// seed (`total_attack / max_pieces` — a top-out keeps its denominator).
fn evaluate(
    width: usize,
    depth: u8,
    weights: &Cc2Weights,
    seeds: &[u64],
    max_pieces: u32,
) -> (Vec<f64>, Vec<MarathonOutcome>) {
    let bot = BotSpec::beam(width, depth).cc2(*weights);
    let stats = evaluate_capped(&bot.factory(), seeds, DEFAULT_MAX_FRAMES, max_pieces);
    let fits = stats
        .outcomes
        .iter()
        .map(|o| f64::from(o.total_attack) / f64::from(max_pieces))
        .collect();
    (fits, stats.outcomes)
}

/// Emit one `games.jsonl` row per game: which arm played (`inc`/`cand`/`val-*`),
/// at which iteration, plus the raw outcome facts.
fn emit_games(arm: &str, iter: u32, outcomes: &[MarathonOutcome]) {
    for o in outcomes {
        events::game(json!({
            "arm": arm,
            "iter": iter,
            "seed": events::seed_hex(o.seed),
            "pieces": o.pieces,
            "topped": o.topped_out,
            "attack": o.total_attack,
        }));
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

pub fn run(spec: &Spec, subject: &Bot, rt: &Runtime) -> io::Result<Value> {
    let (width, depth, origin) = subject_surface(subject)?;
    let campaign = Campaign::derive(spec.campaign);
    let budget = rt.budget(DEFAULT_BUDGET_SECS);
    let start = Instant::now();

    let mut rng = SplitMix64::new(0xA99C_11B0_u64 ^ campaign.slot as u64);
    let origin_vec = to_vec(&origin);
    let scales = origin_vec.map(|x| x.abs().max(0.25));
    let mut cur = origin_vec;
    let mut sigma = spec.sigma0;

    let bar = crate::progress::spinner("app-climb");
    let mut accepts = 0u32;
    let mut iter = 0u32;
    let mut budget_hit = false;
    let mut block_seeds: Vec<u64> = Vec::new();
    let mut inc_fits: Vec<f64> = Vec::new();

    while iter < spec.iters {
        if start.elapsed() >= budget {
            budget_hit = true;
            break;
        }
        // Rotate to a fresh screening block (and re-baseline the incumbent on it).
        if iter.is_multiple_of(spec.block_iters) {
            let block = iter / spec.block_iters;
            block_seeds = seed_set_from(
                campaign.rotation_block(block, spec.seeds_per_block),
                spec.seeds_per_block,
            );
            let (fits, outcomes) = evaluate(
                width,
                depth,
                &to_weights(&origin, &cur),
                &block_seeds,
                spec.max_pieces,
            );
            inc_fits = fits;
            emit_games("inc", iter, &outcomes);
        }

        let cand = perturb(&cur, &scales, sigma, &mut rng);
        let (cand_fits, outcomes) = evaluate(
            width,
            depth,
            &to_weights(&origin, &cand),
            &block_seeds,
            spec.max_pieces,
        );
        emit_games("cand", iter, &outcomes);

        // Paired t-gate on the CRN deltas.
        let deltas: Vec<f64> = cand_fits
            .iter()
            .zip(&inc_fits)
            .map(|(c, i)| c - i)
            .collect();
        let d_mean = mean(&deltas);
        let n = deltas.len() as f64;
        let var = deltas.iter().map(|d| (d - d_mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let se = (var / n).sqrt();
        let accepted = d_mean > spec.accept_k * se;

        if accepted {
            cur = cand;
            inc_fits = cand_fits;
            accepts += 1;
            sigma *= 1.3;
        } else {
            sigma *= spec.sigma_decay;
        }
        sigma = sigma.clamp(spec.sigma_floor, 1.0);
        iter += 1;
        bar.set_message(format!(
            "iter {iter} acc {accepts} σ{sigma:.2} train {:.3}",
            mean(&inc_fits)
        ));
        bar.tick();
    }
    bar.finish_and_clear();

    // Self-validation on the campaign's held-out region: the honest readout.
    let val_seeds = seed_set_from(campaign.validation(spec.val_seeds), spec.val_seeds);
    let (val_origin, o1) = evaluate(width, depth, &origin, &val_seeds, spec.max_pieces);
    emit_games("val-origin", iter, &o1);
    let (val_best, o2) = evaluate(
        width,
        depth,
        &to_weights(&origin, &cur),
        &val_seeds,
        spec.max_pieces,
    );
    emit_games("val-best", iter, &o2);

    let final_weights = to_weights(&origin, &cur);
    eprintln!(
        "app-climb {} @ beam({width},{depth}) cap={} | {iter} iters {accepts} accepts | \
         validation APP {:.4} -> {:.4} ({} held-out seeds){}",
        subject.name,
        spec.max_pieces,
        mean(&val_origin),
        mean(&val_best),
        spec.val_seeds,
        if budget_hit { " | budget hit" } else { "" },
    );

    Ok(json!({
        "iterations": iter,
        "accepts": accepts,
        "budget_hit": budget_hit,
        "sigma_final": sigma,
        "app": { "origin_val": mean(&val_origin), "best_val": mean(&val_best), "train_last": mean(&inc_fits) },
        "board_params": final_weights.board_params().to_vec(),
        "reward_params": final_weights.reward_params().to_vec(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The param vector and the weights must be exact inverses over the
    /// subject's fixed fields — the climb tunes exactly the documented 26 dims.
    #[test]
    fn param_vector_round_trips() {
        let origin = Cc2Weights::attack_tuned();
        let v = to_vec(&origin);
        assert_eq!(to_weights(&origin, &v), origin);

        let mut moved = v;
        moved[0] += 1.0; // a board dim
        moved[BOARD] += 1.0; // the attack dim
        let w = to_weights(&origin, &moved);
        assert_eq!(w.cell_coveredness, origin.cell_coveredness + 1.0);
        assert_eq!(w.attack, origin.attack + 1.0);
        assert_eq!(w.max_cell_covered_height, origin.max_cell_covered_height);
    }

    /// The ES walk is deterministic: same spec, same subject ⇒ the identical
    /// perturbation sequence (the climb-level replay guarantee).
    #[test]
    fn perturbation_walk_is_deterministic() {
        let origin = to_vec(&Cc2Weights::attack_tuned());
        let scales = origin.map(|x| x.abs().max(0.25));
        let walk = || {
            let mut rng = SplitMix64::new(42);
            let mut out = Vec::new();
            for _ in 0..5 {
                out.push(perturb(&origin, &scales, 0.15, &mut rng));
            }
            out
        };
        assert_eq!(walk(), walk());
    }
}
