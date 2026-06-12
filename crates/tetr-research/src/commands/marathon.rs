//! Capped-marathon eval: score/sec + APP for one bot — the tight-loop
//! headline metric (`score_per_second` is the marathon proxy,
//! `attack_per_piece` the versus/CC2 metric; both are autoresearch parse
//! contracts). The piece cap keeps iterations fast; score/sec stays a
//! faithful early-game scoring-rate proxy.
//!
//! # RUN RECORD (2026-06-12, the APP campaign — cap 150, TRAIN / holdout)
//!
//! Goal: APP toward 1.0. Every gain came from SEARCH CLASS on the fixed
//! attack-tuned eval; every eval-side lever was null (climb + 8 probes — see
//! `bots.rs` and the app-climb header). The ladder (TRAIN, 6 seeds):
//! dt20 0.101 → cc2-default 0.409 → attack-tuned(d2) 0.460 → d3 0.572 →
//! d4 0.628 → d6 0.649 → d6w32 0.721 → w64d6 0.738 → bf1k-d8 0.743 →
//! bf2k-d8 0.782 → **tp128d9 0.8256**. Holdout (16 VALIDATION seeds, one
//! read per candidate): d6w32 0.6829, bf2k 0.7721, bf4k 0.7804 (the TRAIN
//! bf2k>bf4k dip was 6-seed noise), **tp128d9 0.8225**
//! (`20260612-083859-marathon-holdout-24880`) — the champion; the codex
//! worktree's 0.8289 claim REPRODUCED under receipts (three disjoint seed
//! sets within ±0.006). Gains are concentration, not combo-farm: attack per
//! line rises 1.55 (d3) → 1.88 (d6w32) → 2.05 (bf2k) at flat ~57 lines/150
//! pieces, i.e. B2B quad/T-spin play. The old ~0.67 eval+search ceiling is
//! SUPERSEDED (it predated the bitboard strike, combo tracking, and the B2B
//! fixes); the eval-optimum ceiling still binds past ~0.83 — beyond it the
//! recorded map says RL/self-play, not tuning.

use serde_json::json;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::events;
use crate::marathon::{DEFAULT_MAX_FRAMES, evaluate_capped};
use crate::seeds::seed_set_from;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub seeds: usize,
    /// Iteration piece cap — part of the metric definition.
    pub max_pieces: u32,
    /// First seed index ([`crate::seeds::seed_set_from`]): 0 is the TRAIN
    /// region (the tight-loop default); holdout entries point into
    /// [`crate::seeds::regions::VALIDATION`] with disjoint offsets.
    pub seed_start: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 6,
            max_pieces: 150,
            seed_start: 0,
        }
    }
}

pub fn run(spec: &Spec, bot: &Bot, _rt: &Runtime) -> std::io::Result<serde_json::Value> {
    let seeds = seed_set_from(spec.seed_start, spec.seeds);
    let stats = evaluate_capped(
        &bot.spec.factory(),
        &seeds,
        DEFAULT_MAX_FRAMES,
        spec.max_pieces,
    );
    for o in &stats.outcomes {
        events::game(json!({
            "seed": events::seed_hex(o.seed),
            "score": o.score,
            "level": o.level,
            "lines": o.lines,
            "pieces": o.pieces,
            "frames": o.frames,
            "topped": o.topped_out,
            "completed": o.completed,
            "attack": o.total_attack,
        }));
    }
    eprintln!(
        "{} cap={} | {} seeds | APP={:.4} attack/game={:.1} | score={:.0} level={:.2} pieces={:.0} completion={:.0}%",
        bot.name,
        spec.max_pieces,
        seeds.len(),
        stats.mean_attack_per_piece,
        stats.mean_attack,
        stats.mean_score,
        stats.mean_level,
        stats.mean_pieces,
        stats.completion_rate * 100.0,
    );
    Ok(json!({
        "score_per_second": stats.mean_score_per_second,
        "attack_per_piece": stats.mean_attack_per_piece,
    }))
}
