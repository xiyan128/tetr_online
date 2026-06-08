//! The attack-strike metrics suite: **APP** (attack per piece — the primary metric),
//! **DS/P** (downstack per piece — lines cleared per piece under garbage), survival
//! rate, and a behavior breakdown (clear-type histogram, T-spins, B2B, combo, PCs).
//!
//! Everything is measured across garbage **scenarios** so APP is probed under
//! realistic pressure, not just on the gameable empty board. Each scenario is a pure
//! function of the bot factory + seed (same engine seed ⇒ same pieces; seeded garbage
//! holes), so results are reproducible and drop straight into the autoresearch loop.
//!
//! Combo is advanced on **line clears only** (see [`crate::action_clear_lines`]) — a
//! hard drop emits its own `ScoreAwarded`, which must not inflate combo/attack.

use std::time::Instant;

use tetr_core::engine::{
    CellKind, Engine, EngineEvent, EngineScoreAction, PieceType, TSpinKind,
};
use tetr_core::player::{drive_engine, PlayerController};

use crate::{
    action_clear_lines, cheese_holes, controller_seed, fold_combo, marathon_config, versus_hole,
    GarbageQueue, MAX_PIECE_FRAMES,
};

/// A garbage scenario to measure the bot in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    /// Empty board, `max_pieces` budget — the offense ceiling. NOTE: combo-gameable;
    /// read it alongside the pressured scenarios, never alone.
    Clean { max_pieces: u32 },
    /// Start buried under `rows` of seeded cheese; play until cleared or `max_pieces`.
    /// Probes digging (DS/P) and attack-while-digging.
    Cheese { rows: u32, max_pieces: u32 },
    /// Sustained pressure: every `period` pieces the faucet queues `lines` of seeded
    /// garbage; the bot's attack cancels pending first, the remainder dumps. Pressure
    /// = `lines / period` lines per piece. Probes survival + APP under load.
    Faucet {
        period: u32,
        lines: u32,
        max_pieces: u32,
    },
}

impl Scenario {
    fn max_pieces(self) -> u32 {
        match self {
            Scenario::Clean { max_pieces }
            | Scenario::Cheese { max_pieces, .. }
            | Scenario::Faucet { max_pieces, .. } => max_pieces,
        }
    }

    /// A short label for reports.
    pub fn label(self) -> String {
        match self {
            Scenario::Clean { max_pieces } => format!("clean/{max_pieces}p"),
            Scenario::Cheese { rows, max_pieces } => format!("cheese{rows}/{max_pieces}p"),
            Scenario::Faucet {
                period,
                lines,
                max_pieces,
            } => format!("faucet{lines}per{period}/{max_pieces}p"),
        }
    }
}

/// The canonical scenario set: offense ceiling, digging, and two pressure levels.
pub fn standard_suite() -> Vec<Scenario> {
    vec![
        Scenario::Clean { max_pieces: 100 },
        Scenario::Cheese {
            rows: 9,
            max_pieces: 60,
        },
        Scenario::Faucet {
            period: 4,
            lines: 1,
            max_pieces: 100,
        }, // light pressure ≈ 0.25 lines/piece
        Scenario::Faucet {
            period: 2,
            lines: 1,
            max_pieces: 100,
        }, // heavy pressure ≈ 0.5 lines/piece
    ]
}

/// Per-game behavior + attack breakdown.
#[derive(Clone, Debug, Default)]
pub struct BehaviorStats {
    pub pieces: u32,
    /// Attack SENT (offense). In a Faucet this is post-cancellation (lines that
    /// actually reach the opponent after eating incoming garbage).
    pub attack: u32,
    /// Total lines cleared (≈ downstack volume under garbage).
    pub lines: u32,
    pub topped_out: bool,
    // clear-type histogram
    pub singles: u32,
    pub doubles: u32,
    pub triples: u32,
    pub tetrises: u32,
    pub tspin_mini: u32,
    pub tspin_single: u32,
    pub tspin_double: u32,
    pub tspin_triple: u32,
    pub perfect_clears: u32,
    // chain / combo
    pub b2b_clears: u32,
    /// Highest combo counter reached (0-based: first clear = 0, second = 1, …).
    pub max_combo: u32,
    /// Clears that earned a combo bonus (the 2nd-and-later clear in a chain).
    pub combo_clears: u32,
    // pressure
    pub garbage_received: u32,
}

impl BehaviorStats {
    /// Attack Per Piece — the primary strike metric.
    pub fn app(&self) -> f32 {
        self.attack as f32 / self.pieces.max(1) as f32
    }
    /// Downstack per piece: lines cleared per piece (digging rate under garbage).
    pub fn dsp(&self) -> f32 {
        self.lines as f32 / self.pieces.max(1) as f32
    }
    /// Attack per line cleared — high ⇒ concentrated (Tetris/T-spin), low ⇒ combo-spam.
    pub fn attack_per_line(&self) -> f32 {
        self.attack as f32 / self.lines.max(1) as f32
    }

    fn record_clear(&mut self, action: EngineScoreAction, b2b_bonus: bool, combo: u32, pc: bool) {
        match action {
            EngineScoreAction::Single => self.singles += 1,
            EngineScoreAction::Double => self.doubles += 1,
            EngineScoreAction::Triple => self.triples += 1,
            EngineScoreAction::Tetris => self.tetrises += 1,
            EngineScoreAction::TSpin { kind, lines } => match (kind, lines) {
                (TSpinKind::Mini, _) => self.tspin_mini += 1,
                (TSpinKind::Full, 1) => self.tspin_single += 1,
                (TSpinKind::Full, 2) => self.tspin_double += 1,
                (TSpinKind::Full, 3) => self.tspin_triple += 1,
                _ => {}
            },
            _ => {}
        }
        if b2b_bonus {
            self.b2b_clears += 1;
        }
        if pc {
            self.perfect_clears += 1;
        }
        if combo >= 1 {
            self.combo_clears += 1;
        }
        self.max_combo = self.max_combo.max(combo);
    }
}

/// Play one game of `scenario` and return its behavior stats.
pub fn play_scenario(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    scenario: Scenario,
) -> BehaviorStats {
    let max_pieces = scenario.max_pieces();
    let mut engine = Engine::new(marathon_config(), seed);

    // Initial cheese for the Cheese scenario (each row full except its seeded hole).
    if let Scenario::Cheese { rows, .. } = scenario {
        for (y, &hole) in cheese_holes(seed, rows as usize).iter().enumerate() {
            for x in 0..10isize {
                if x as usize != hole {
                    engine.set_cell(x, y as isize, CellKind::Some(PieceType::I));
                }
            }
        }
    }

    let mut bot = make_bot(controller_seed(seed));
    let mut stats = BehaviorStats::default();
    let mut combo = 0u32;
    let mut pending = GarbageQueue::default();
    let mut hole_rng = seed ^ 0xF00D_BABE_1234_5678;
    let mut frames = 0u32;
    let max_frames = max_pieces.saturating_mul(MAX_PIECE_FRAMES).max(50_000);

    while stats.pieces < max_pieces && frames < max_frames {
        frames += 1;
        let mut locked = false;
        for event in drive_engine(&mut engine, &mut *bot) {
            // Combo + clear accounting lives in `fold_combo` (the single home of the
            // combo-gating rule); the faucet cancellation is this scenario's own layer.
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                stats.record_clear(
                    clear.action,
                    clear.back_to_back_bonus,
                    clear.combo,
                    clear.perfect_clear,
                );
                stats.lines += action_clear_lines(clear.action) as u32;
                // Offense: under a faucet, attack cancels pending garbage first.
                let sent = if matches!(scenario, Scenario::Faucet { .. }) {
                    pending.cancel(clear.attack)
                } else {
                    clear.attack
                };
                stats.attack += sent;
            }
            match &event {
                EngineEvent::Locked { .. } => {
                    locked = true;
                    stats.pieces += 1;
                }
                EngineEvent::GameOver { .. } => stats.topped_out = true,
                _ => {}
            }
        }
        if stats.topped_out {
            break;
        }

        // Faucet tick: dump whatever survived cancellation, then queue the next batch
        // (giving the bot `period` pieces to cancel it before it lands).
        if let Scenario::Faucet { period, lines, .. } = scenario {
            if locked && period > 0 && stats.pieces % period == 0 {
                let received = pending.pending();
                if received > 0 {
                    stats.garbage_received += received;
                    if pending.dump(&mut engine) {
                        stats.topped_out = true;
                        break;
                    }
                }
                pending.push(lines, versus_hole(&mut hole_rng));
            }
        }

        // Cheese: stop once the garbage has been dug out.
        if let Scenario::Cheese { rows, .. } = scenario {
            if locked && engine.snapshot().lines as u32 >= rows {
                break;
            }
        }
    }

    stats
}

/// Aggregate stats for a scenario over a seed set.
#[derive(Clone, Debug)]
pub struct ScenarioReport {
    pub scenario: Scenario,
    pub games: usize,
    pub survival_rate: f32,
    pub mean_app: f32,
    pub mean_dsp: f32,
    pub mean_attack_per_line: f32,
    pub mean_pieces: f32,
    pub mean_garbage_received: f32,
    /// Mean wall-clock per piece (ms) over the seed set — the **compute axis** of the
    /// compute/quality frontier. Times the full per-piece loop (dominated by the
    /// planner's search); meaningful only on an unloaded machine (no concurrent jobs).
    pub mean_ms_per_piece: f32,
    /// Summed clear-type counts across all games (the behavior histogram).
    pub totals: BehaviorStats,
}

/// Run `scenario` over `seeds` and aggregate.
pub fn evaluate_scenario(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    scenario: Scenario,
) -> ScenarioReport {
    let t0 = Instant::now();
    let games: Vec<BehaviorStats> = seeds
        .iter()
        .map(|&s| play_scenario(make_bot, s, scenario))
        .collect();
    let elapsed_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;
    let n = games.len().max(1) as f32;
    let survived = games.iter().filter(|g| !g.topped_out).count();
    let total_pieces: u64 = games.iter().map(|g| g.pieces as u64).sum();

    let mut totals = BehaviorStats::default();
    for g in &games {
        totals.pieces += g.pieces;
        totals.attack += g.attack;
        totals.lines += g.lines;
        totals.singles += g.singles;
        totals.doubles += g.doubles;
        totals.triples += g.triples;
        totals.tetrises += g.tetrises;
        totals.tspin_mini += g.tspin_mini;
        totals.tspin_single += g.tspin_single;
        totals.tspin_double += g.tspin_double;
        totals.tspin_triple += g.tspin_triple;
        totals.perfect_clears += g.perfect_clears;
        totals.b2b_clears += g.b2b_clears;
        totals.combo_clears += g.combo_clears;
        totals.garbage_received += g.garbage_received;
        totals.max_combo = totals.max_combo.max(g.max_combo);
    }

    ScenarioReport {
        scenario,
        games: games.len(),
        survival_rate: survived as f32 / n,
        mean_app: games.iter().map(|g| g.app()).sum::<f32>() / n,
        mean_dsp: games.iter().map(|g| g.dsp()).sum::<f32>() / n,
        mean_attack_per_line: games.iter().map(|g| g.attack_per_line()).sum::<f32>() / n,
        mean_pieces: games.iter().map(|g| g.pieces as f32).sum::<f32>() / n,
        mean_garbage_received: games.iter().map(|g| g.garbage_received as f32).sum::<f32>() / n,
        mean_ms_per_piece: elapsed_ms / total_pieces.max(1) as f32,
        totals,
    }
}
