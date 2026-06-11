//! The engine's plain-data contract: configuration in, events and snapshots
//! out. Everything here is behavior-free data every host (game, embed,
//! research) imports; the machine that produces it lives in [`api`](super::api).

use crate::engine::garbage::GarbageBatch;
use crate::engine::goals::GoalSystem;
use crate::engine::pieces::{PieceRotation, PieceType};
use crate::engine::scoring::EngineScoreAction;
use crate::engine::{LockDownMode, LOCK_DOWN_SECONDS, MIN_LEVEL};

/// Hidden rows above the visible field — the guideline buffer zone where
/// pieces spawn and can lock (§16.4). A constant, not a config knob: nothing
/// ever varied it, and the engine's rules (spawn rows, lock-out) assume it.
pub const BUFFER_HEIGHT: usize = 20;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub board_width: usize,
    pub visible_height: usize,
    pub preview_count: usize,
    pub lock_down_mode: LockDownMode,
    pub lock_down_seconds: f32,
    pub starting_level: u8,
    pub goal_system: GoalSystem,
    /// Versus: the maximum pending-garbage lines that rise onto the board after
    /// one clear-less lock (the "garbage cap"). Pending lines beyond the cap
    /// stay queued for the next opportunity. `0` disables rising entirely —
    /// pending garbage can then only ever be cancelled (a config edge, not the
    /// "uncapped" convention some games use). Irrelevant outside versus — the
    /// queue is only fed by [`Engine::queue_garbage`].
    pub garbage_cap: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            board_width: 10,
            visible_height: 20,
            preview_count: 5,
            lock_down_mode: LockDownMode::Extended,
            lock_down_seconds: LOCK_DOWN_SECONDS,
            starting_level: MIN_LEVEL,
            goal_system: GoalSystem::Fixed,
            garbage_cap: 8,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputFrame {
    pub dt_seconds: f32,
    pub left: bool,
    pub right: bool,
    pub soft_drop: bool,
    pub hard_drop: bool,
    pub rotate_clockwise: bool,
    pub rotate_counterclockwise: bool,
    pub hold: bool,
    pub pause: bool,
}

/// Game-facing happenings of one [`Engine::step`]. Deliberately NOT a movement
/// trace: spawning and per-cell movement (lateral, soft-drop, gravity) are
/// snapshot state — observe them by diffing [`EngineSnapshot::active`] across
/// steps. Events exist for the discrete outcomes a consumer cannot recover
/// from a snapshot diff alone (locks, scores, attack, game over).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Rotated {
        piece_type: PieceType,
        rotation: PieceRotation,
        origin: (isize, isize),
        kick_number: u8,
    },
    HardDropped {
        piece_type: PieceType,
        cells_dropped: usize,
    },
    Locked {
        piece_type: PieceType,
        lines_cleared: usize,
    },
    ScoreAwarded {
        action: EngineScoreAction,
        score: usize,
        total_score: usize,
        back_to_back_bonus: bool,
    },
    Held {
        held: PieceType,
        active: PieceType,
    },
    /// Versus: this lock's attack survived cancellation — `lines` garbage lines
    /// leave the board for the opponent (net of any pending garbage it offset;
    /// a fully-cancelled attack emits nothing). The match driver routes this to
    /// the opponent's [`Engine::queue_garbage`].
    AttackSent {
        lines: u32,
    },
    /// Versus: pending garbage rose onto the board after a clear-less lock
    /// (`lines` rows, capped per lock by [`EngineConfig::garbage_cap`]).
    GarbageInserted {
        lines: u32,
    },
    GameOver {
        reason: GameOverStatus,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GameOverStatus {
    BlockOut,
    LockOut,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SnapshotCell {
    pub x: isize,
    pub y: isize,
    /// The colour identity of the cell. For a garbage cell this is a legacy
    /// fill (`I`) kept so colour-by-piece consumers keep working; check
    /// [`garbage`](Self::garbage) first — a versus renderer paints garbage
    /// neutral, not cyan.
    pub piece_type: PieceType,
    /// True for a garbage-row cell ([`CellKind::Garbage`]); always `false` for
    /// active-piece and ghost cells.
    pub garbage: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActivePieceSnapshot {
    pub piece_type: PieceType,
    pub rotation: PieceRotation,
    pub origin: (isize, isize),
    pub cells: Vec<SnapshotCell>,
    pub hold_used: bool,
    pub landed: bool,
    pub lock_timer_seconds: f32,
    pub lock_timer_fraction: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    pub config: EngineConfig,
    pub board_cells: Vec<SnapshotCell>,
    pub active: Option<ActivePieceSnapshot>,
    pub ghost_cells: Vec<SnapshotCell>,
    pub hold: Option<PieceType>,
    pub next_queue: Vec<PieceType>,
    pub score: usize,
    pub lines: usize,
    pub level: u8,
    pub goal_remaining: usize,
    pub back_to_back_active: bool,
    /// Consecutive line-clearing placements so far (the combo counter; `0` when no
    /// combo is active). Lets a search resume from the real in-game combo instead of
    /// assuming `0`, so it can value continuing a chain.
    pub combo: u32,
    /// Pieces the generator's **current 7-bag** has not yet dealt — the exact set
    /// the next piece *beyond the revealed queue* draws from (empty ⇒ a bag
    /// boundary: the next deal opens a fresh bag of all seven). Exported because a
    /// search speculating past the queue cannot reconstruct this from `next_queue`
    /// alone: the queue window straddles bag boundaries, so any reconstruction is
    /// wrong whenever the active piece is not the first piece of its bag.
    pub bag_remainder: Vec<PieceType>,
    /// Versus: the pending-garbage queue against this player, oldest batch
    /// first (empty outside versus). Hole columns are already determined (drawn
    /// at queue time from this engine's seeded stream), so a search can model
    /// rising exactly; [`pending_garbage_total`](Self::pending_garbage_total)
    /// is the incoming-meter sum a UI shows.
    pub pending_garbage: Vec<GarbageBatch>,
    pub game_over: Option<GameOverStatus>,
}

impl EngineSnapshot {
    /// Total pending-garbage lines (the incoming meter). Saturating, like the
    /// engine's own meter.
    pub fn pending_garbage_total(&self) -> u32 {
        self.pending_garbage
            .iter()
            .map(|b| b.lines)
            .fold(0u32, u32::saturating_add)
    }
}
