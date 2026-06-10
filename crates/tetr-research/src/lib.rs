//! Headless **marathon scoring-speed** evaluation for the Tetris bot.
//!
//! Bevy-free: depends only on `tetr-core` (engine + AI seam), so it compiles and
//! runs fast enough for an iterative hill-climb loop. The metric is **score per
//! simulated second** in Marathon mode
//! (`GoalSystem::Variable`, end at `MAX_LEVEL`), measured at `Handicap::perfect()`
//! so it reflects *policy quality*, not the in-game reaction handicap.
//!
//! Determinism: a game is a pure function of `(bot factory, seed)` — the engine's
//! 7-bag and the policy RNG are both seeded. Re-running an evaluation reproduces
//! every number.

use std::time::Duration;

use std::collections::VecDeque;

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights, Evaluator, LinearEvaluator, Weights};
use tetr_core::ai::{
    AiController, BeamPlanner, BestFirstPlanner, Handicap, Policy, SearchBudget, SearchPolicy,
};
use tetr_core::engine::{
    attack_lines, CellKind, Engine, EngineConfig, EngineEvent, EngineScoreAction, EngineSnapshot,
    GoalSystem, InputFrame, PieceType, MAX_LEVEL,
};
use tetr_core::player::{drive_engine, PlayerController};

/// TBP client for baselining Cold Clear 2 as a subprocess. See [`cc2`].
pub mod cc2;

/// APP / DS-P / behavior metrics across garbage scenarios — the attack-strike suite.
pub mod behavior;

/// Shared env-config + deterministic-RNG helpers for the research `bin/` tools.
pub mod cli;

use cli::SplitMix64;

/// Fixed simulation rate: one engine step (one `drive_engine` poll) = 1/60 s.
pub const SIM_HZ: f32 = 60.0;

/// Generous per-game frame cap (≈ 4.6 hours of sim time) so a stalling bot can
/// never hang the harness. Marathon normally ends far sooner (level 15 or top-out).
pub const DEFAULT_MAX_FRAMES: u32 = 1_000_000;

/// Engine configuration for a Marathon game: the Variable goal system (what
/// `Variant::Marathon` applies) on the default board. Marathon ends when the
/// snapshot level reaches `MAX_LEVEL`.
pub fn marathon_config() -> EngineConfig {
    EngineConfig {
        goal_system: GoalSystem::Variable,
        ..EngineConfig::default()
    }
}

/// Derive the controller RNG seed from the game seed (decorrelated from the
/// engine's piece stream, but fully determined by it — matches the arena harness).
pub(crate) fn controller_seed(seed: u64) -> u64 {
    seed ^ 0x9E37_79B9_7F4A_7C15
}

/// Lines a scoring action actually cleared (0 for drops / no-clear / a 0-line spin).
///
/// The combo counter must advance on **line clears only**. A hard drop emits its own
/// `ScoreAwarded { action: HardDrop }` (engine `api.rs`), so a loop that bumps combo
/// on every `ScoreAwarded` would inflate combo (and thus attack) by ~1 per piece.
/// Gate combo + attack on `action_clear_lines(action) > 0`.
pub(crate) fn action_clear_lines(action: EngineScoreAction) -> usize {
    match action {
        EngineScoreAction::Single => 1,
        EngineScoreAction::Double => 2,
        EngineScoreAction::Triple => 3,
        EngineScoreAction::Tetris => 4,
        EngineScoreAction::TSpin { lines, .. } => lines,
        EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. }
        | EngineScoreAction::NoClear => 0,
    }
}

/// One line clear's accounting, produced by [`fold_combo`].
pub(crate) struct ClearInfo {
    pub action: EngineScoreAction,
    pub back_to_back_bonus: bool,
    pub perfect_clear: bool,
    /// Combo index used for this clear (pre-increment; `0` for the first in a chain).
    pub combo: u32,
    /// Garbage lines this clear sends ([`attack_lines`] with the pre-clear combo).
    pub attack: u32,
}

/// Fold one engine event into the running `combo`, returning the clear it produced (if
/// any). The single home for combo/attack accounting: combo advances on line clears
/// only — a hard drop emits its own `ScoreAwarded` that must NOT bump it — and resets
/// on a clear-less lock.
/// Callers still do their own piece counting / top-out / stats from the same event.
pub(crate) fn fold_combo(
    event: &EngineEvent,
    engine: &Engine,
    combo: &mut u32,
) -> Option<ClearInfo> {
    match event {
        EngineEvent::Locked { lines_cleared, .. } => {
            if *lines_cleared == 0 {
                *combo = 0; // a non-clearing placement breaks the chain
            }
            None
        }
        EngineEvent::ScoreAwarded {
            action,
            back_to_back_bonus,
            ..
        } if action_clear_lines(*action) > 0 => {
            // Post-clear board: empty ⇒ perfect clear. Cheap: no snapshot alloc.
            let perfect_clear = engine.board_is_empty();
            let index = *combo;
            let attack = attack_lines(*action, *back_to_back_bonus, index, perfect_clear);
            *combo += 1;
            Some(ClearInfo {
                action: *action,
                back_to_back_bonus: *back_to_back_bonus,
                perfect_clear,
                combo: index,
                attack,
            })
        }
        _ => None,
    }
}

/// The result of one Marathon game.
#[derive(Debug, Clone, Copy)]
pub struct MarathonOutcome {
    pub seed: u64,
    pub score: u32,
    pub level: u8,
    pub lines: u32,
    pub pieces: u32,
    pub frames: u32,
    pub topped_out: bool,
    /// Reached `MAX_LEVEL` without topping out (a "won" marathon).
    pub completed: bool,
    /// Total garbage lines sent (guideline attack table) over the game — the
    /// versus-relevant quantity. `attack_per_piece()` is the APP efficiency metric
    /// we compare against Cold Clear 2.
    pub total_attack: u32,
}

impl MarathonOutcome {
    pub fn elapsed_seconds(&self) -> f32 {
        self.frames as f32 / SIM_HZ
    }

    /// The headline marathon metric: score accumulated per simulated second.
    pub fn score_per_second(&self) -> f32 {
        let t = self.elapsed_seconds();
        if t > 0.0 {
            self.score as f32 / t
        } else {
            0.0
        }
    }

    /// Attack per piece (APP): garbage lines sent ÷ pieces placed — the standard
    /// offensive-efficiency metric for comparing versus bots (vs Cold Clear 2).
    pub fn attack_per_piece(&self) -> f32 {
        if self.pieces > 0 {
            self.total_attack as f32 / self.pieces as f32
        } else {
            0.0
        }
    }
}

/// Play one Marathon game to completion (`MAX_LEVEL`), top-out, the frame cap, or
/// `max_pieces` placements — the fast `/autoresearch` metric path. A piece cap turns
/// a full ~930-piece marathon into a quick proxy while keeping score/sec meaningful
/// as an early-game scoring rate. Pass `u32::MAX` for the full, uncapped marathon
/// (final validation). `make_bot` builds a fresh controller per (controller) seed,
/// so games stay independent and reproducible.
pub fn play_marathon_capped(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    max_frames: u32,
    max_pieces: u32,
) -> MarathonOutcome {
    let mut engine = Engine::new(marathon_config(), seed);
    let mut bot = make_bot(controller_seed(seed));

    let mut pieces = 0u32;
    let mut frames = 0u32;
    let mut topped = false;
    // Versus attack accounting (guideline table). `combo` = consecutive
    // line-clearing placements; `total_attack` sums garbage lines sent.
    let mut combo = 0u32;
    let mut total_attack = 0u32;

    while frames < max_frames {
        frames += 1;
        let mut locked = false;
        for event in drive_engine(&mut engine, &mut *bot) {
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                total_attack += clear.attack;
            }
            match &event {
                EngineEvent::Locked { .. } => {
                    pieces += 1;
                    locked = true;
                }
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped {
            break;
        }
        // Level only rises on a line clear (at a lock), so checking then suffices
        // and avoids a full snapshot every frame.
        if locked && engine.snapshot().level >= MAX_LEVEL {
            break;
        }
        // Fast-metric cap: stop after a bounded number of placements (u32::MAX = off).
        if pieces >= max_pieces {
            break;
        }
    }

    let snap = engine.snapshot();
    MarathonOutcome {
        seed,
        score: snap.score as u32,
        level: snap.level,
        lines: snap.lines as u32,
        pieces,
        frames,
        topped_out: topped,
        completed: snap.level >= MAX_LEVEL && !topped,
        total_attack,
    }
}

/// Aggregate statistics over a seed set.
#[derive(Debug, Clone)]
pub struct MarathonStats {
    pub games: usize,
    pub mean_score_per_second: f32,
    pub mean_score: f32,
    pub mean_level: f32,
    pub mean_pieces: f32,
    pub completion_rate: f32,
    pub topout_rate: f32,
    /// Mean attack per piece (APP) — the versus offensive-efficiency metric.
    pub mean_attack_per_piece: f32,
    /// Mean total attack (garbage lines sent) per game.
    pub mean_attack: f32,
    pub outcomes: Vec<MarathonOutcome>,
}

/// Evaluate a bot over `seeds`, returning aggregate Marathon stats.
pub fn evaluate(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    max_frames: u32,
) -> MarathonStats {
    evaluate_capped(make_bot, seeds, max_frames, u32::MAX)
}

/// Like [`evaluate`] but with a per-game `max_pieces` cap — the fast metric path
/// the `/autoresearch` loop uses (full uncapped marathon = `u32::MAX`).
pub fn evaluate_capped(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    max_frames: u32,
    max_pieces: u32,
) -> MarathonStats {
    let outcomes: Vec<MarathonOutcome> = seeds
        .iter()
        .map(|&seed| play_marathon_capped(make_bot, seed, max_frames, max_pieces))
        .collect();

    let n = outcomes.len().max(1) as f32;
    let sum = |f: &dyn Fn(&MarathonOutcome) -> f32| outcomes.iter().map(f).sum::<f32>();

    MarathonStats {
        games: outcomes.len(),
        mean_score_per_second: sum(&|o| o.score_per_second()) / n,
        mean_score: sum(&|o| o.score as f32) / n,
        mean_level: sum(&|o| o.level as f32) / n,
        mean_pieces: sum(&|o| o.pieces as f32) / n,
        completion_rate: sum(&|o| if o.completed { 1.0 } else { 0.0 }) / n,
        topout_rate: sum(&|o| if o.topped_out { 1.0 } else { 0.0 }) / n,
        mean_attack_per_piece: sum(&|o| o.attack_per_piece()) / n,
        mean_attack: sum(&|o| o.total_attack as f32) / n,
        outcomes,
    }
}

// --- Downstack (cheese) benchmark --------------------------------------------
// Clearing seeded garbage rows efficiently tests digging / board-reading — the
// skill that separates elite versus bots — and, unlike empty-board APP, is NOT
// gameable by combo-farming. Fewer pieces to clear the cheese = stronger.

/// Garbage-hole column per row for a seeded cheese board (independent per row =
/// maximum messiness). Both bots face the identical cheese for a given seed.
pub fn cheese_holes(seed: u64, rows: usize) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    (0..rows).map(|_| (rng.next_u64() % 10) as usize).collect()
}

/// One cheese-clear game's result.
#[derive(Debug, Clone, Copy)]
pub struct DownstackOutcome {
    pub seed: u64,
    pub garbage_rows: u32,
    /// Pieces placed until the cheese was cleared (or the cap / top-out hit).
    pub pieces: u32,
    /// Cleared `garbage_rows` lines without topping out.
    pub cleared: bool,
    pub topped_out: bool,
    /// Attack sent while digging (guideline table) — the OFFENSE proxy: a bot that
    /// clears garbage as Tetrises / T-spins / B2B counter-attacks; one that clears
    /// sloppily sends ~0. Higher = better, given a comparable pieces-to-clear.
    pub attack: u32,
}

/// Play one cheese-clear game: start with `garbage_rows` seeded garbage rows and
/// measure how many pieces the bot needs to clear that many lines (fewer = better).
pub fn play_downstack(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    garbage_rows: u32,
    max_pieces: u32,
) -> DownstackOutcome {
    let mut engine = Engine::new(EngineConfig::default(), seed);
    // Paint the cheese: each row full except its hole column. (`set_cell` is the
    // engine's board-setup seam; the piece colour is irrelevant to line clears.)
    for (y, &hole) in cheese_holes(seed, garbage_rows as usize).iter().enumerate() {
        for x in 0..10isize {
            if x as usize != hole {
                engine.set_cell(x, y as isize, CellKind::Some(PieceType::I));
            }
        }
    }

    let mut bot = make_bot(controller_seed(seed));
    let mut pieces = 0u32;
    let mut frames = 0u32;
    let mut topped = false;
    let mut combo = 0u32;
    let mut total_attack = 0u32;
    let max_frames = max_pieces.saturating_mul(64).max(10_000);

    while frames < max_frames {
        frames += 1;
        let mut locked = false;
        for event in drive_engine(&mut engine, &mut *bot) {
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                total_attack += clear.attack;
            }
            match &event {
                EngineEvent::Locked { .. } => {
                    pieces += 1;
                    locked = true;
                }
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped {
            break;
        }
        // Lines only rise at a lock; the cheese is cleared once we've cleared that many.
        if locked && engine.snapshot().lines as u32 >= garbage_rows {
            break;
        }
        if pieces >= max_pieces {
            break;
        }
    }

    let cleared = engine.snapshot().lines as u32 >= garbage_rows && !topped;
    DownstackOutcome {
        seed,
        garbage_rows,
        pieces,
        cleared,
        topped_out: topped,
        attack: total_attack,
    }
}

/// Aggregate downstack stats over a seed set.
#[derive(Debug, Clone)]
pub struct DownstackStats {
    pub games: usize,
    /// Mean pieces to clear the cheese, over games that cleared it (DEFENSE).
    pub mean_pieces_to_clear: f32,
    /// Mean attack sent while clearing, over games that cleared it (OFFENSE proxy).
    pub mean_attack: f32,
    pub clear_rate: f32,
    pub outcomes: Vec<DownstackOutcome>,
}

/// Evaluate a bot's cheese-clear efficiency over `seeds`.
pub fn evaluate_downstack(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    garbage_rows: u32,
    max_pieces: u32,
) -> DownstackStats {
    let outcomes: Vec<DownstackOutcome> = seeds
        .iter()
        .map(|&seed| play_downstack(make_bot, seed, garbage_rows, max_pieces))
        .collect();
    let cleared: Vec<&DownstackOutcome> = outcomes.iter().filter(|o| o.cleared).collect();
    let (mean_pieces_to_clear, mean_attack) = if cleared.is_empty() {
        (0.0, 0.0)
    } else {
        let n = cleared.len() as f32;
        (
            cleared.iter().map(|o| o.pieces as f32).sum::<f32>() / n,
            cleared.iter().map(|o| o.attack as f32).sum::<f32>() / n,
        )
    };
    DownstackStats {
        games: outcomes.len(),
        mean_pieces_to_clear,
        mean_attack,
        clear_rate: cleared.len() as f32 / outcomes.len().max(1) as f32,
        outcomes,
    }
}

// --- Versus (mutual garbage) benchmark ---------------------------------------
// The complete head-to-head: both bots play the identical piece sequence and a
// player loses by topping out. The garbage RULES (cancellation, rising after
// clear-less locks, the per-lock cap, hole choice) are the ENGINE's — see
// tetr-core's garbage module — and `play_versus` only routes each side's net
// AttackSent to the other side's queue. The `GarbageQueue` below is a HARNESS
// scheduler kept for the scripted pressure scenarios (behavior.rs Faucet) and
// the TBP referee (`cc2_baseline`), which inserts raw and does its own
// bookkeeping. NOTE one deliberate divergence: this harness queue settles the
// OLDEST garbage lowest (drain_newest_first), while the engine's chronological
// rising leaves the NEWEST batch lowest — boards from the two paths are not
// comparable for multi-batch deliveries, and `cc2_baseline` win rates are not
// like-for-like with `play_versus` (the referee also keeps the old wholesale
// dump timing). This is Cold Clear 2's home turf — the metric that actually
// decides "beat CC2", as opposed to one-sided downstacking.

/// Frames a single piece may take before we treat the bot as wedged (~4.3s at 60 Hz
/// — far beyond any real per-piece search, so only a genuinely stuck bot trips it).
pub(crate) const MAX_PIECE_FRAMES: u32 = 256;

/// Garbage queued against a player: a FIFO of `(lines, hole_col)` batches, one per
/// un-cancelled opponent attack. Your own clears cancel the oldest batches first;
/// whatever you fail to cancel is dumped onto your board.
#[derive(Default)]
pub struct GarbageQueue {
    batches: VecDeque<(u32, usize)>,
}

impl GarbageQueue {
    /// Total garbage lines currently queued.
    pub fn pending(&self) -> u32 {
        self.batches.iter().map(|&(n, _)| n).sum()
    }

    pub fn push(&mut self, lines: u32, hole: usize) {
        if lines > 0 {
            self.batches.push_back((lines, hole));
        }
    }

    /// Cancel up to `attack` lines from the front; return the un-cancelled remainder.
    pub fn cancel(&mut self, mut attack: u32) -> u32 {
        while attack > 0 {
            let Some(front) = self.batches.front_mut() else {
                break;
            };
            let c = attack.min(front.0);
            front.0 -= c;
            attack -= c;
            if front.0 == 0 {
                self.batches.pop_front();
            }
        }
        attack
    }

    /// Remove all queued batches, newest first — so a caller inserting them one by
    /// one (each landing at the bottom) settles the oldest garbage lowest.
    pub fn drain_newest_first(&mut self) -> Vec<(u32, usize)> {
        let mut out = Vec::with_capacity(self.batches.len());
        while let Some(batch) = self.batches.pop_back() {
            out.push(batch);
        }
        out
    }

    /// Dump all queued garbage onto `engine` (newest batch first, see above).
    /// Returns true if the rising stack tops the player out.
    pub fn dump(&mut self, engine: &mut Engine) -> bool {
        let mut topped = false;
        for (lines, hole) in self.drain_newest_first() {
            topped |= engine.insert_garbage(lines as usize, hole);
        }
        topped
    }
}

/// Next seeded garbage-hole column (SplitMix64 over a per-match stream).
pub fn versus_hole(rng: &mut u64) -> usize {
    // Thread the caller's bare `u64` state through the shared SplitMix64 step: one
    // `next_u64` advances the word exactly as the inlined fold did, then write it back.
    let mut gen = SplitMix64::from_raw(*rng);
    let hole = (gen.next_u64() % 10) as usize;
    *rng = gen.into_raw();
    hole
}

/// Drive one player's bot until it locks a single piece (or tops out / stalls).
/// Returns `(net attack sent by that placement, topped_out)`.
///
/// Attack accounting is the **engine's**: [`EngineEvent::AttackSent`] already
/// carries the post-cancellation net (the engine offsets its own pending queue
/// at lock time), and pending garbage rises by the engine's guideline timing —
/// after a clear-less lock, capped per lock. The caller's only job is routing
/// the net attack to the opponent's queue. (When nothing was ever queued the
/// pending queue is empty and net == gross — which is how the TBP referee path
/// keeps its own external bookkeeping.)
fn versus_step_piece(engine: &mut Engine, bot: &mut dyn PlayerController) -> (u32, bool) {
    let mut attack = 0u32;
    let mut topped = false;
    for _ in 0..MAX_PIECE_FRAMES {
        let mut locked = false;
        for event in drive_engine(engine, bot) {
            match &event {
                EngineEvent::AttackSent { lines } => attack += lines,
                EngineEvent::Locked { .. } => locked = true,
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped || locked {
            break;
        }
    }
    (attack, topped)
}

/// One side of a versus match: an engine + its bot. Exposed so an external referee
/// (e.g. the Cold Clear 2 driver, which runs the opponent over TBP) can pit our bot
/// against another protocol bot using the same garbage rules as [`play_versus`].
pub struct VersusEngine {
    engine: Engine,
    bot: Box<dyn PlayerController>,
}

impl VersusEngine {
    pub fn new(make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>, seed: u64) -> Self {
        Self {
            engine: Engine::new(marathon_config(), seed),
            bot: make_bot(controller_seed(seed)),
        }
    }

    /// Place one piece; return `(attack produced, topped_out)`. The referee
    /// inserts garbage raw ([`receive`](Self::receive)), so this engine's
    /// pending queue stays empty and the attack reported here is gross — the
    /// referee does its own cancellation bookkeeping externally.
    pub fn step_piece(&mut self) -> (u32, bool) {
        versus_step_piece(&mut self.engine, &mut *self.bot)
    }

    /// Receive one garbage batch (`lines` rows, hole at `hole_col`); return true if
    /// it tops this player out.
    pub fn receive(&mut self, lines: u32, hole_col: usize) -> bool {
        self.engine.insert_garbage(lines as usize, hole_col)
    }
}

/// Result of a single versus match (A = first bot, B = second).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersusResult {
    AWins,
    BWins,
    Draw,
}

/// Decide a versus match from each side's final state. A topout loses outright — a
/// death-loss takes **priority** over attack count; otherwise the larger attack total
/// wins and equal totals draw. Shared by [`run_versus`] and the CC2 referee harness
/// (`cc2-baseline`) so both score matches identically. `A` is the first side, `B` the
/// second (e.g. in the CC2 harness A = ours, B = CC2).
pub fn decide_versus(a_topped: bool, b_topped: bool, a_attack: u32, b_attack: u32) -> VersusResult {
    use std::cmp::Ordering;
    match (a_topped, b_topped) {
        (true, false) => VersusResult::BWins, // A died
        (false, true) => VersusResult::AWins, // B died
        // Both alive, or both dead the same tick: the bigger attacker wins, ties draw.
        _ => match a_attack.cmp(&b_attack) {
            Ordering::Greater => VersusResult::AWins,
            Ordering::Less => VersusResult::BWins,
            Ordering::Equal => VersusResult::Draw,
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VersusOutcome {
    pub seed: u64,
    pub result: VersusResult,
    pub plies: u32,
    pub attack_a: u32,
    pub attack_b: u32,
    pub a_topped: bool,
    pub b_topped: bool,
}

/// Salt folding the match seed into the HARNESS-side garbage-hole RNG used by the
/// TBP referee (`cc2_baseline`), decorrelating hole placement from the
/// (same-seeded) piece stream. `play_versus` no longer draws holes here at all:
/// engine-rules matches use each receiver engine's own internal stream
/// (tetr-core's garbage module, its own salt).
pub const VERSUS_HOLE_SALT: u64 = 0xA5A5_5A5A_DEAD_BEEF;

/// Play one versus match between bot A and bot B. Both face the identical piece
/// sequence (same engine seed), so the result reflects decision quality, not piece
/// luck. A player loses by topping out; if the ply cap is reached with both alive,
/// the higher total attack wins.
///
/// "Attack" here is **net** attack — the post-cancellation spillover that actually
/// lands on the opponent's queue (the standard "garbage sent" notion), not gross
/// lines generated. Lines spent cancelling your own incoming queue count for
/// survival but not for this tiebreaker, so a pure digging bot under pressure can
/// record zero attack while playing well.
pub fn play_versus(
    make_a: &dyn Fn(u64) -> Box<dyn PlayerController>,
    make_b: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    max_plies: u32,
) -> VersusOutcome {
    // Level rises but never ends the game here (only top-out / the cap do).
    // The versus rules — cancellation, rising after clear-less locks, the
    // garbage cap, hole choice — are the ENGINE's (see tetr-core's garbage
    // module); this driver only routes each side's net attack to the other
    // side's pending queue.
    let mut a_engine = Engine::new(marathon_config(), seed);
    let mut b_engine = Engine::new(marathon_config(), seed);
    let mut a_bot = make_a(controller_seed(seed));
    let mut b_bot = make_b(controller_seed(seed));
    let (mut a_attack, mut b_attack) = (0u32, 0u32);
    let (mut a_topped, mut b_topped) = (false, false);
    let mut plies = 0u32;

    'match_loop: for ply in 0..max_plies {
        // Alternate first mover so neither side gets a structural send-first edge.
        let order = if ply % 2 == 0 { [0u8, 1] } else { [1, 0] };
        for &who in &order {
            plies += 1;
            // Route the attack BEFORE checking death: the engine already
            // encodes the rule (a lock-out lock emits no AttackSent), so any
            // attack that WAS emitted — e.g. a real clear whose next spawn
            // block-outs — legitimately left the board and must reach the
            // opponent's queue and the stats. The driver never second-guesses
            // the event stream.
            if who == 0 {
                let (atk, topped) = versus_step_piece(&mut a_engine, &mut *a_bot);
                if atk > 0 {
                    b_engine.queue_garbage(atk);
                    a_attack += atk;
                }
                if topped {
                    a_topped = true;
                    break 'match_loop;
                }
            } else {
                let (atk, topped) = versus_step_piece(&mut b_engine, &mut *b_bot);
                if atk > 0 {
                    a_engine.queue_garbage(atk);
                    b_attack += atk;
                }
                if topped {
                    b_topped = true;
                    break 'match_loop;
                }
            }
        }
    }

    let result = decide_versus(a_topped, b_topped, a_attack, b_attack);

    VersusOutcome {
        seed,
        result,
        plies,
        attack_a: a_attack,
        attack_b: b_attack,
        a_topped,
        b_topped,
    }
}

/// Aggregate versus stats over a seed set.
#[derive(Debug, Clone)]
pub struct VersusStats {
    pub games: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
    pub mean_attack_a: f32,
    pub mean_attack_b: f32,
    pub outcomes: Vec<VersusOutcome>,
}

impl VersusStats {
    pub fn a_win_rate(&self) -> f32 {
        self.a_wins as f32 / self.games.max(1) as f32
    }
}

/// Evaluate bot A vs bot B over `seeds`.
pub fn evaluate_versus(
    make_a: &dyn Fn(u64) -> Box<dyn PlayerController>,
    make_b: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    max_plies: u32,
) -> VersusStats {
    let outcomes: Vec<VersusOutcome> = seeds
        .iter()
        .map(|&seed| play_versus(make_a, make_b, seed, max_plies))
        .collect();
    let a_wins = outcomes
        .iter()
        .filter(|o| o.result == VersusResult::AWins)
        .count();
    let b_wins = outcomes
        .iter()
        .filter(|o| o.result == VersusResult::BWins)
        .count();
    let draws = outcomes.len() - a_wins - b_wins;
    let n = outcomes.len().max(1) as f32;
    let mean_attack_a = outcomes.iter().map(|o| o.attack_a as f32).sum::<f32>() / n;
    let mean_attack_b = outcomes.iter().map(|o| o.attack_b as f32).sum::<f32>() / n;
    VersusStats {
        games: outcomes.len(),
        a_wins,
        b_wins,
        draws,
        mean_attack_a,
        mean_attack_b,
        outcomes,
    }
}

/// A controller wrapper that hides the pending-garbage queue from its inner
/// bot: the snapshot it forwards has `pending_garbage` emptied, so the bot
/// plans as if no attack were queued — the *blind* arm of the
/// garbage-awareness A/B. Everything else (weights, search, venue, seeds) stays
/// identical, so a win-rate gap between a wrapped and an unwrapped copy of the
/// same bot measures exactly the value of seeing (and modeling) the queue.
pub struct BlindToGarbage(pub Box<dyn PlayerController>);

impl PlayerController for BlindToGarbage {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
        if snapshot.pending_garbage.is_empty() {
            return self.0.poll(snapshot); // nothing to hide: skip the clone
        }
        let mut blinded = snapshot.clone();
        blinded.pending_garbage.clear();
        self.0.poll(&blinded)
    }
}

/// A deterministic, well-distributed set of `count` seeds (SplitMix64 over indices).
pub fn seed_set(count: usize) -> Vec<u64> {
    seed_set_from(0, count)
}

/// Like [`seed_set`] but over indices `start..start+count` — for **disjoint**
/// train / held-out validation seed sets (`seed_set(n)` and `seed_set_from(s, n)`
/// share no seeds when `s >= n`), so a hillclimb can be checked for overfitting.
pub fn seed_set_from(start: usize, count: usize) -> Vec<u64> {
    // Per-index SplitMix64 seeding: `new(i).next_u64()` reproduces the old inline fold
    // (`new` stores `i`, then `next_u64` folds in the golden increment) bit-for-bit.
    (start as u64..(start + count) as u64)
        .map(|i| SplitMix64::new(i).next_u64())
        .collect()
}

// --- Bot factories -----------------------------------------------------------

/// The current shipped bot: greedy search over the linear DT-20 / SURVIVAL
/// evaluator, at full strength (`Handicap::perfect()`). This is the baseline.
pub fn baseline_bot(seed: u64) -> Box<dyn PlayerController> {
    Box::new(AiController::new(Handicap::perfect(), seed))
}

/// Core beam-bot constructor: a deterministic [`BeamPlanner`] over `eval` at full
/// strength (imperfection 0, no reaction delay — measures pure policy quality). Every
/// beam contender funnels through here, so the planner / budget / strength convention
/// lives in one place and head-to-heads stay apples-to-apples. Adding a new contender
/// is one line: `beam_bot(seed, w, d, Box::new(MyEvaluator::new(..)))`.
pub fn beam_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    let policy = SearchPolicy::new(
        Box::new(BeamPlanner::new(beam_width)),
        eval,
        SearchBudget::beam(max_depth),
        0.0, // no imperfection — measure policy quality
        seed,
    );
    Box::new(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        Duration::ZERO,
    ))
}

/// Core **best-first-search** bot: a [`BestFirstPlanner`] over `eval` at full strength
/// (imperfection 0, no reaction delay). The best-first analogue of [`beam_bot`] with
/// the SAME eval/strength convention, so a head-to-head isolates the **search
/// algorithm** — the beam's fixed-width generations vs best-first's node-budgeted
/// graph search with transposition. `node_budget` is total expansions per decision;
/// `max_depth` caps lookahead plies.
pub fn bestfirst_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    let policy = SearchPolicy::new(
        Box::new(BestFirstPlanner::new()),
        eval,
        SearchBudget::best_first(node_budget, max_depth),
        0.0, // no imperfection — measure policy quality
        seed,
    );
    Box::new(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        Duration::ZERO,
    ))
}

/// A best-first bot over CC2's evaluator with custom [`Cc2Weights`] — the search-
/// algorithm counterpart of [`beam_cc2_weights_bot`], for an apples-to-apples
/// best-first-vs-beam comparison at a fixed eval.
pub fn bestfirst_cc2_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    bestfirst_bot(
        seed,
        node_budget,
        max_depth,
        Box::new(Cc2Evaluator::new(weights)),
    )
}

/// A best-first bot over the linear evaluator with explicit [`Weights`] — the
/// counterpart of [`beam_weights_bot`]. Pairs best-first's deep-line search with the
/// `near_full_rows` combo feature, to test whether it can find the clean-board combo
/// cascade the beam's fixed-width truncation prunes.
pub fn bestfirst_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    bestfirst_bot(
        seed,
        node_budget,
        max_depth,
        Box::new(LinearEvaluator::new(weights)),
    )
}

/// The Tier-2 beam bot: a deterministic `BeamPlanner` over the **same** linear
/// DT-20 / SURVIVAL evaluator the baseline uses, at full strength (imperfection 0,
/// no reaction delay). It differs from [`baseline_bot`] in *only* the planner
/// (greedy → beam), so a head-to-head isolates the search depth's effect on
/// score/sec. `beam_width` controls truncation; `max_depth` the lookahead plies
/// (`max_depth == 1` reproduces the greedy decision exactly — the seam-faithful
/// gate). Bag speculation past the visible queue is on (the `BeamPlanner` default).
pub fn beam_linear_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(LinearEvaluator::default()),
    )
}

/// **Cold Clear 2's evaluator, ported** ([`Cc2Evaluator`]) on our beam — CC2's
/// *evaluation function* playing on our engine and search. Identical planner,
/// budget, and strength to [`beam_linear_bot`]; only the evaluator differs, so a
/// head-to-head isolates eval quality. Crucially this plays the **fair** versus
/// harness on our engine with real garbage — the comparison the TBP bridge could
/// not give (CC2 has no garbage message). This is the baseline to hillclimb past.
pub fn beam_cc2_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(Cc2Evaluator::default()),
    )
}

/// A beam bot over an explicit linear [`Weights`] set — lets a head-to-head vary
/// the board features and/or the reward profile on the same planner/strength (e.g.
/// DT-20 board + Cold-Clear *concentrated-attack* reward vs the shipped SURVIVAL
/// reward that cashes every clear).
pub fn beam_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(LinearEvaluator::new(weights)),
    )
}

/// Like [`beam_cc2_bot`] but with **custom** CC2 weights — the hillclimb's
/// candidate factory. Only the evaluator's weights differ.
pub fn beam_cc2_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(Cc2Evaluator::new(weights)),
    )
}

#[cfg(test)]
mod versus_rules_tests {
    use super::*;
    use tetr_core::ai::{AiController, Handicap};

    /// THE accounting gate for moving attack into the engine: over a real bot
    /// game with nothing queued (pending empty ⇒ net == gross), the engine's
    /// AttackSent events must total exactly what the research-side fold
    /// (`fold_combo` + `attack_lines`, the convention every APP baseline was
    /// recorded under) computes from the same event stream.
    #[test]
    fn engine_attack_events_match_the_research_fold() {
        let mut engine = Engine::new(marathon_config(), 11);
        let mut bot = AiController::new(Handicap::perfect(), 99);
        let mut combo = 0u32;
        let (mut fold_total, mut event_total) = (0u32, 0u32);
        for _ in 0..4_000 {
            if engine.snapshot().game_over.is_some() {
                break;
            }
            for event in drive_engine(&mut engine, &mut bot) {
                if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                    fold_total += clear.attack;
                }
                if let EngineEvent::AttackSent { lines } = event {
                    event_total += lines;
                }
            }
        }
        assert!(fold_total > 0, "the bot must have attacked at least once");
        assert_eq!(
            event_total, fold_total,
            "engine-side attack must reproduce the research fold bit-for-bit"
        );
    }

    /// A whole match is a pure function of its seed: same seed, same bots ⇒
    /// identical outcome (the property SPRT and win-rate climbs rely on).
    #[test]
    fn play_versus_is_deterministic() {
        let make = |seed: u64| -> Box<dyn PlayerController> {
            Box::new(AiController::new(Handicap::perfect(), seed))
        };
        let run = || {
            let o = play_versus(&make, &make, 42, 40);
            (
                o.result, o.plies, o.attack_a, o.attack_b, o.a_topped, o.b_topped,
            )
        };
        assert_eq!(run(), run());
    }
}

#[cfg(test)]
mod versus_decision_tests {
    use super::{decide_versus, VersusResult};

    #[test]
    fn topout_loses_before_attack_is_compared() {
        // A dies with *more* attack dealt → B still wins: a death-loss takes priority.
        assert_eq!(decide_versus(true, false, 100, 0), VersusResult::BWins);
        assert_eq!(decide_versus(false, true, 0, 100), VersusResult::AWins);
    }

    #[test]
    fn both_alive_higher_attack_wins_and_ties_draw() {
        assert_eq!(decide_versus(false, false, 5, 3), VersusResult::AWins);
        assert_eq!(decide_versus(false, false, 3, 5), VersusResult::BWins);
        assert_eq!(decide_versus(false, false, 4, 4), VersusResult::Draw);
    }

    #[test]
    fn double_topout_falls_back_to_attack_dealt() {
        // Both topped the same tick: decide by attack landed before dying.
        assert_eq!(decide_versus(true, true, 7, 2), VersusResult::AWins);
        assert_eq!(decide_versus(true, true, 2, 2), VersusResult::Draw);
    }
}
