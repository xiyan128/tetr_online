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
//! # RUN RECORD (2026-06-12, the watch arm — budgeted scan + line commitment)
//!
//! The node-grain/commitment refactor's gate, all on the 4-seed opener
//! screen (receipts recorded from the implementation tree, pre-commit):
//! pc-reveal-s28w8 replays its recorded games **byte-identically** (PPC
//! 0.0875, `20260612-223949-pc-opener-screen-v1-87254` ==
//! `20260612-212339`'s games.jsonl) — the registered arms cross the refactor
//! untouched. The in-game operating point **pc-watch-v1** (cap 14 × width 2,
//! scan_node_budget 60k, commit_lines, depth-3 fallback) reads **0.025 PPC /
//! 0.5625 APP** (`20260612-224554-…-94325`) vs the pre-budget in-game arm
//! probe-pc-watch-unbudgeted's 0.0125 / 0.4250 (`20260612-224657-…-95275`)
//! — twice the PC rate at ~1/100 the compute (4 seeds finish in ~3 s vs the
//! budget truncating the unbudgeted arm). probe-pc-watch-uncommitted reads
//! the same 0.025 (`20260612-224652-…-95078`): at this sample size the
//! budget cut does the work and the commitment is PPC-neutral — it buys the
//! zero-search follow-ups (in-game smoothness), not coverage.
//!
//! CAVEATS: 4–8 TRAIN seeds, ≤7 PCs per arm — screening signal only.
//! `pc-validation-v1` (held-out) has NOT been read; read it once per promoted
//! candidate.
//!
//! # RUN RECORD (2026-06-12, Layer 4 — shared-prefix scan)
//!
//! Layer 4 searches the visible queue once and forks per scenario at the first
//! unknown draw (`PcCoverageConfig::shared_prefix`). Registered research arms
//! keep `shared_prefix: false` so recorded games stay byte-identical; the
//! interactive arm (`pc-watch-v1`, `shared_prefix: true`) trades a fresh PPC
//! screen for ~2× headless scan throughput on the opener eval.
//!
//! With `shared_prefix: true` on the registered arm (exploratory only):
//! PPC **0.0625** (5 PCs / 80, `20260612-231116-…-24793`) vs the pre-Layer-4
//! **0.0875** — canonical-tail transposition during the prefix changes which
//! futures survive truncation, so this is expected until a per-scenario-tail
//! prefix (true byte match) is implemented. The game arm **pc-watch-v1** with
//! Layer 4 reads **0.0375 PPC** (`20260612-231337-…-27678`) vs **0.025**
//! without — a win at the interactive operating point despite the research-arm
//! regression on the exploratory run.

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
