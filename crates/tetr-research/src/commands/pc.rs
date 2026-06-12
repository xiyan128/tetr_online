//! Clean-board perfect-clear eval: PPC (perfect clears per piece) for one bot.
//!
//! PPC aggregates as total engine-true PCs over total pieces; APP and top-out
//! rate ride along so failed PC attempts stay visible. Per-game events include
//! the piece index of every PC, so one eval kind answers both the sustained
//! and the opener question. Games materialize in seed order; the wall-clock
//! budget truncates only **between** games, so a budget-cut run is an honest
//! prefix of the unbounded one.
//!
//! # RUN RECORD (2026-06-12, the PC campaign — worktree screen, reproduced)
//!
//! Goal: PPC on a clean board. The control says reward shaping is NOT enough:
//! in the codex-worktree screening, general search with a dominating PC reward
//! sat at 0.0100–0.0125 PPC (tp128d9, bf2k-pc40, and tp256d12 with
//! perfect_clear=1000+override — TRAIN, 8×100 pieces). Scenario-coverage
//! search moves it. REPRODUCED here under receipts on the 4-seed opener
//! screen: **pc-reveal-s28w8 0.0875 PPC** (7 PCs / 80 pieces, APP 1.05,
//! `20260612-212339-pc-opener-screen-v1-4726`) and pc-scenario-s8w2 0.0500
//! (`20260612-212848-pc-opener-screen-v1-8832`) — both play the worktree's
//! recorded games seed-for-seed, so its wider ladder transfers: reveal
//! coverage 0.075 (s14w8) / 0.075 (s56w8), and 0.0340 on the 100-piece screen
//! (s14w2) vs tp128d9's 0.0125. Probed and rejected there: PC-shape ranking
//! (0.025), reveal+mass tiebreak (0.0625), per-reveal robustness (0.0625) —
//! dropped from the planner (its header carries the record). Cost notes: the
//! s28w8 opener read costs ~80 s/game release — the default 180 s budget
//! truncates it, pass `--budget-secs 420` for all four seeds; the worktree's
//! scenario_cap×width=256 arm could not finish ONE 100-piece game in 300 s.
//!
//! CAVEATS: 4–8 TRAIN seeds, ≤7 PCs per arm — screening signal only.
//! `pc-validation-v1` (held-out) has NOT been read; read it once per promoted
//! candidate.

use std::time::Instant;

use serde_json::json;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::events;
use crate::pc::{play_pc_capped, summarize};
use crate::seeds::seed_set_from;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub seeds: usize,
    /// Per-game piece cap — part of the metric definition (PPC is only
    /// comparable at one cap: opener-heavy policies inflate at small caps).
    pub max_pieces: u32,
    /// First seed index ([`crate::seeds::seed_set_from`]); screens point at
    /// TRAIN, the holdout entry at a VALIDATION offset disjoint from the
    /// marathon holdouts.
    pub seed_start: usize,
    /// Default wall-clock bound (`--budget-secs` overrides). Coverage arms
    /// vary ~50× in cost, so each entry documents its own.
    pub default_budget_secs: u64,
}

pub fn run(spec: &Spec, bot: &Bot, rt: &Runtime) -> std::io::Result<serde_json::Value> {
    let seeds = seed_set_from(spec.seed_start, spec.seeds);
    let budget = rt.budget(spec.default_budget_secs);
    let started = Instant::now();
    let mut outcomes = Vec::with_capacity(seeds.len());

    for seed in seeds {
        // Always finish the first game (an empty run verifies nothing), then
        // stop at the budget between games: an honest prefix.
        if !outcomes.is_empty() && started.elapsed() >= budget {
            break;
        }
        outcomes.push(play_pc_capped(&bot.spec.factory(), seed, spec.max_pieces));
    }

    let stats = summarize(outcomes);
    for o in &stats.outcomes {
        events::game(json!({
            "seed": events::seed_hex(o.seed),
            "pieces": o.pieces,
            "perfect_clears": o.perfect_clears,
            "pc_piece_indices": o.pc_piece_indices,
            "attack": o.total_attack,
            "frames": o.frames,
            "topped": o.topped_out,
        }));
    }

    let complete = stats.games == spec.seeds;
    eprintln!(
        "{} cap={} | {}/{} seeds{} | PC={} pieces={} PPC={:.6} | APP={:.4} topout={:.1}%",
        bot.name,
        spec.max_pieces,
        stats.games,
        spec.seeds,
        if complete { "" } else { " (budget prefix)" },
        stats.perfect_clears,
        stats.pieces,
        stats.pc_per_piece(),
        stats.attack_per_piece(),
        stats.topout_rate() * 100.0,
    );

    Ok(json!({
        "complete": complete,
        "games": stats.games,
        "pieces": stats.pieces,
        "perfect_clears": stats.perfect_clears,
        "pc_per_piece": stats.pc_per_piece(),
        "attack_per_piece": stats.attack_per_piece(),
        "topout_rate": stats.topout_rate(),
    }))
}
