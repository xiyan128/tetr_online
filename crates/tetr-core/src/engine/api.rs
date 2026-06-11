//! The public engine facade: [`Engine`], its config, inputs, events, and
//! snapshots.
//!
//! This is the one type the host (the Bevy layer) talks to. [`Engine::step`]
//! advances the simulation by one [`InputFrame`] and returns the [`EngineEvent`]s
//! that occurred; [`Engine::snapshot`] produces a self-contained [`EngineSnapshot`]
//! for rendering. The engine owns the board, the active piece, the seven-bag
//! generator, hold slot, and score/goal state, and wires together the pure rule
//! modules in this crate. Per ADR-7 the engine carries no rendering or Bevy
//! types; it is driven entirely through these plain data structures.

use crate::engine::active_piece::ActivePiece;
use crate::engine::attack::attack_lines;
use crate::engine::board::{Board, CellKind};
use crate::engine::game_over::{is_block_out, is_lock_out};
use crate::engine::garbage::PendingGarbage;
use crate::engine::generator::PieceGenerator;
use crate::engine::gravity::fall_speed_seconds;
use crate::engine::lock_clear::lock_and_clear;
use crate::engine::lock_down::apply_grounded_move_or_rotation;
use crate::engine::pieces::{MoveDirection, Piece, PieceRotation, PieceType};
use crate::engine::scoring::{score_action, EngineScoreAction, ScoreAward, ScoreState};
use crate::engine::t_spin::{classify_t_spin, is_t_slot, TSpinKind};
use crate::engine::types::*;
use crate::engine::RotationDirection;

pub struct Engine {
    config: EngineConfig,
    board: Board,
    active: Option<ActivePiece>,
    generator: PieceGenerator,
    next_queue: Vec<PieceType>,
    hold: Option<PieceType>,
    score_state: ScoreState,
    game_over: Option<GameOverStatus>,
    gravity_accumulator_seconds: f32,
    /// Versus: incoming garbage queued against this player (see `garbage.rs`).
    garbage: PendingGarbage,
}

impl Engine {
    pub fn new(config: EngineConfig, seed: u64) -> Self {
        let board =
            Board::with_top_margin(config.board_width, config.visible_height, BUFFER_HEIGHT);
        let score_state = ScoreState::new(config.goal_system, config.starting_level);
        let mut engine = Self {
            config,
            board,
            active: None,
            generator: PieceGenerator::with_seed(seed),
            next_queue: Vec::new(),
            hold: None,
            score_state,
            game_over: None,
            gravity_accumulator_seconds: 0.0,
            garbage: PendingGarbage::new(seed),
        };
        engine.fill_next_queue();
        engine
    }

    pub fn step(&mut self, input: InputFrame) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        if self.game_over.is_some() {
            return events;
        }

        if self.active.is_none() {
            self.spawn_next_piece(&mut events);
        }
        if self.game_over.is_some() {
            return events;
        }

        if input.hold {
            self.hold_active_piece(&mut events);
        }
        if self.game_over.is_some() {
            return events;
        }

        if input.hard_drop {
            self.hard_drop_active_piece(&mut events);
            return events;
        }

        if input.rotate_clockwise {
            self.rotate_active_piece(RotationDirection::Clockwise, &mut events);
        } else if input.rotate_counterclockwise {
            self.rotate_active_piece(RotationDirection::Counterclockwise, &mut events);
        }

        match (input.left, input.right) {
            (true, false) => self.move_active_piece(MoveDirection::Left, &mut events),
            (false, true) => self.move_active_piece(MoveDirection::Right, &mut events),
            _ => {}
        }

        if input.soft_drop {
            self.move_active_piece(MoveDirection::Down, &mut events);
        }

        self.advance_time(input.dt_seconds.max(0.0), &mut events);

        events
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            config: self.config.clone(),
            board_cells: self.board_snapshot_cells(),
            active: self
                .active
                .as_ref()
                .map(|active| active_piece_snapshot(active, &self.config)),
            ghost_cells: self.ghost_snapshot_cells(),
            hold: self.hold,
            next_queue: self.next_queue.clone(),
            score: self.score_state.score(),
            lines: self.score_state.lines(),
            level: self.score_state.level(),
            goal_remaining: self.score_state.goal_remaining(),
            back_to_back_active: self.score_state.back_to_back_active(),
            combo: self.score_state.combo(),
            bag_remainder: self.generator.bag_remainder().to_vec(),
            pending_garbage: self.garbage.batches().collect(),
            game_over: self.game_over,
        }
    }

    /// True iff the playfield is empty — the perfect-clear test. Cheap: delegates to
    /// [`Board::is_empty`], which short-circuits and allocates nothing, so sim loops
    /// can check it per line clear without building a full [`snapshot`](Self::snapshot).
    pub fn board_is_empty(&self) -> bool {
        self.board.is_empty()
    }

    /// Board-setup seam: paint a single board cell, bypassing the per-frame loop.
    ///
    /// Returns `true` if `(x, y)` is inside the board. Exists so the acceptance
    /// suite (`tests/acceptance_*.rs`) and the research harness (`tetr-research`'s
    /// crafted cheese boards) can construct board preconditions that are not
    /// reachable deterministically through [`Engine::step`] alone — the same thing
    /// the in-crate unit tests do via the private `board` field. It only forwards
    /// to [`Board::set`]; it adds no rule behavior of its own. Hidden from docs
    /// because it is a harness seam, not part of the game-facing API.
    #[doc(hidden)]
    pub fn set_cell(&mut self, x: isize, y: isize, cell: CellKind) -> bool {
        self.board.set(x, y, cell)
    }

    /// Raise the stack by `count` garbage rows (each row full except `hole_col`) —
    /// the versus-garbage mechanic. Returns `true` if this tops the player out: the
    /// stack is forced past the ceiling, or the active piece is now buried. On a
    /// top-out the engine drops the (buried) active piece and latches
    /// [`GameOverStatus::BlockOut`] so the next [`step`](Self::step) is a no-op —
    /// the same end state the spawn-collision path leaves. Because this mutation
    /// runs out-of-band of [`step`](Self::step) there is no event sink: the `bool`
    /// return (and the latched game-over in the snapshot) is the caller's signal,
    /// not an [`EngineEvent::GameOver`].
    /// QUARANTINED legacy seam: inserts raw, bypassing the pending queue,
    /// cancellation, the cap, and event emission that [`queue_garbage`]
    /// (Self::queue_garbage) owns. Kept ONLY for the TBP referee and the
    /// scripted-pressure scenarios (`tetr-research::versus_legacy`, whose
    /// recorded CC2 baselines it underpins — deleting this means re-recording
    /// them on the engine path first). Everything else uses `queue_garbage`.
    pub fn insert_garbage(&mut self, count: usize, hole_col: usize) -> bool {
        // A finished game accepts no more garbage: the board stays a faithful
        // record of how it ended, and the latched game-over reason is never
        // rewritten post-mortem. "You are (already) topped out" is the honest
        // return.
        if self.game_over.is_some() {
            return true;
        }
        let overflow = self.board.insert_garbage_lines(count, hole_col);
        let buried = self
            .active
            .as_ref()
            .is_some_and(|active| active.piece().collide_with(&self.board, active.origin()));
        if overflow || buried {
            self.active = None;
            self.game_over = Some(GameOverStatus::BlockOut);
        }
        overflow || buried
    }

    /// Versus: queue an opponent's attack of `lines` against this player. The
    /// garbage does not touch the board yet — it sits pending (visible as
    /// [`EngineSnapshot::pending_garbage`]) where this player's own attack can
    /// still cancel it line-for-line, and rises after a lock that clears no
    /// lines (capped per lock by [`EngineConfig::garbage_cap`], emitting
    /// [`EngineEvent::GarbageInserted`]). Each queued batch draws one hole
    /// column from this engine's own seeded stream, so a `(seed, attack
    /// sequence)` reproduces the board exactly. Like
    /// [`insert_garbage`](Self::insert_garbage) this runs out-of-band of
    /// [`step`](Self::step) — queueing has no immediate board effect, so there
    /// is no event to miss.
    pub fn queue_garbage(&mut self, lines: u32) {
        // A finished game accepts no more attack (and must not advance the
        // hole stream): the final snapshot stays a faithful record.
        if self.game_over.is_some() {
            return;
        }
        self.garbage.queue(lines, self.config.board_width);
    }

    /// Test-only seam: install `active` as the current active piece, bypassing
    /// spawn (and its immediate gravity drop). Lets the acceptance suite isolate
    /// behavior for a hand-placed piece. Adds no rule behavior of its own.
    #[doc(hidden)]
    pub fn set_active(&mut self, active: ActivePiece) {
        self.active = Some(active);
    }

    /// Test-only seam: lock a hand-placed `active` through the real
    /// lock/clear/score path ([`Engine::lock_active_piece`]) and return the
    /// emitted events. This is the exact path the per-frame loop uses on
    /// hard-drop/lock-down; it merely takes the piece explicitly instead of
    /// reading `self.active`. Adds no rule behavior of its own.
    #[doc(hidden)]
    pub fn lock_active_for_test(&mut self, active: ActivePiece) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        self.lock_active_piece(active, &mut events);
        events
    }

    /// Test-only seam: rewind the goal/level progression to the starting level,
    /// preserving accumulated score, lines, and the Back-to-Back chain. The
    /// acceptance suite uses this to reproduce the §13.3 scoring example's
    /// explicit "At Level 1" precondition across the full canonical chain, which
    /// clears more than one level's worth of lines (a real Fixed-goal game would
    /// level up after 10 clears per §14.2). Adds no rule behavior of its own.
    #[doc(hidden)]
    pub fn reset_level_for_test(&mut self) {
        self.score_state.reset_level_for_test();
    }

    fn fill_next_queue(&mut self) {
        let target_len = self.config.preview_count.max(1);
        while self.next_queue.len() < target_len {
            self.next_queue
                .push(self.generator.next().expect("piece generator is infinite"));
        }
    }

    fn pop_next_piece_type(&mut self) -> PieceType {
        self.fill_next_queue();
        let piece_type = self.next_queue.remove(0);
        self.fill_next_queue();
        piece_type
    }

    fn spawn_next_piece(&mut self, events: &mut Vec<EngineEvent>) {
        let piece_type = self.pop_next_piece_type();
        self.spawn_piece_type(piece_type, false, events);
    }

    fn spawn_piece_type(
        &mut self,
        piece_type: PieceType,
        hold_used: bool,
        events: &mut Vec<EngineEvent>,
    ) {
        let piece = Piece::from(piece_type);
        let spawn_origin = piece.spawn_coords(self.config.board_width, self.config.visible_height);
        if is_block_out(&piece, &self.board, spawn_origin) {
            self.active = None;
            self.game_over = Some(GameOverStatus::BlockOut);
            events.push(EngineEvent::GameOver {
                reason: GameOverStatus::BlockOut,
            });
            return;
        }

        let origin = piece
            .try_move(&self.board, spawn_origin, MoveDirection::Down)
            .unwrap_or(spawn_origin);
        let mut active = ActivePiece::new(piece_type, origin);
        if hold_used {
            active.mark_hold_used();
        }
        update_landing_state(&self.board, &self.config, &mut active, false, false);
        self.active = Some(active);
        self.gravity_accumulator_seconds = 0.0;
    }

    fn hold_active_piece(&mut self, events: &mut Vec<EngineEvent>) {
        let Some(active) = self.active.as_ref() else {
            return;
        };
        if active.hold_used_on_this_piece() {
            return;
        }

        let outgoing = active.piece_type();
        let incoming = self.hold.replace(outgoing);
        let next_active = incoming.unwrap_or_else(|| self.pop_next_piece_type());
        self.spawn_piece_type(next_active, true, events);
        if self.game_over.is_none() {
            events.push(EngineEvent::Held {
                held: outgoing,
                active: next_active,
            });
        }
    }

    fn move_active_piece(&mut self, direction: MoveDirection, events: &mut Vec<EngineEvent>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let was_landed = active.landed();
        let Some(origin) = active
            .piece()
            .try_move(&self.board, active.origin(), direction)
        else {
            return;
        };

        let action = match direction {
            MoveDirection::Down => crate::engine::PieceAction::SoftDrop,
            MoveDirection::Left | MoveDirection::Right => crate::engine::PieceAction::Move,
        };
        active.move_to(origin, action);
        update_landing_state(
            &self.board,
            &self.config,
            active,
            was_landed,
            matches!(direction, MoveDirection::Left | MoveDirection::Right),
        );
        if direction == MoveDirection::Down {
            self.gravity_accumulator_seconds = 0.0;
            self.score(EngineScoreAction::SoftDrop, events);
        }
    }

    fn rotate_active_piece(&mut self, direction: RotationDirection, events: &mut Vec<EngineEvent>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let was_landed = active.landed();
        let target_rotation = match direction {
            RotationDirection::Clockwise => active.rotation() + PieceRotation::R90,
            RotationDirection::Counterclockwise => active.rotation() + PieceRotation::R270,
        };
        let Some((rotation, origin, kick_number)) =
            active
                .piece()
                .try_rotate_with_kicks(&self.board, active.origin(), target_rotation)
        else {
            return;
        };
        if kick_number == 0 {
            return;
        }

        // §7.5 point-5 override: if SRS test 5 placed a T into a T-slot, set the
        // sticky flag so the spin classifies Full and survives later non-rotation
        // actions (§12.4). Evaluate the slot on the *post-rotation* pose against
        // the current board (the piece is not yet locked), using a throwaway probe
        // so we can compute the flag before committing the real rotation.
        let entered_t_slot_with_kick_5 =
            kick_number == 5 && active.piece_type() == PieceType::T && {
                let mut probe = active.clone();
                probe.rotate_to(rotation, origin, direction, kick_number, false);
                is_t_slot(&probe, &self.board)
            };

        active.rotate_to(
            rotation,
            origin,
            direction,
            kick_number,
            entered_t_slot_with_kick_5,
        );
        update_landing_state(&self.board, &self.config, active, was_landed, true);
        events.push(EngineEvent::Rotated {
            piece_type: active.piece_type(),
            rotation,
            origin,
            kick_number,
        });
    }

    fn hard_drop_active_piece(&mut self, events: &mut Vec<EngineEvent>) {
        let Some(mut active) = self.active.take() else {
            return;
        };
        let mut cells_dropped = 0;
        while let Some(origin) =
            active
                .piece()
                .try_move(&self.board, active.origin(), MoveDirection::Down)
        {
            active.move_to(origin, crate::engine::PieceAction::HardDrop);
            cells_dropped += 1;
        }

        events.push(EngineEvent::HardDropped {
            piece_type: active.piece_type(),
            cells_dropped,
        });
        self.score(
            EngineScoreAction::HardDrop {
                cells: cells_dropped,
            },
            events,
        );
        self.lock_active_piece(active, events);
    }

    fn lock_active_piece(&mut self, active: ActivePiece, events: &mut Vec<EngineEvent>) {
        let piece_type = active.piece_type();
        // Classify the t-spin and lock-out against the pre-lock board/piece
        // state, before `lock_and_clear` mutates the board.
        let t_spin = classify_t_spin(&active, &self.board);
        let lock_out = is_lock_out(active.piece(), active.origin(), self.config.visible_height);

        let outcome = lock_and_clear(&active, &mut self.board);
        let lines_cleared = outcome.cleared_rows.len();

        events.push(EngineEvent::Locked {
            piece_type,
            lines_cleared,
        });
        // The attack table reads the combo index BEFORE this clear advances it
        // (the same pre-increment convention the research harness pinned), so
        // capture it before scoring mutates the chain state.
        let combo_before = self.score_state.combo();
        let award = self.score_lock_result(t_spin, lines_cleared, events);

        if lock_out {
            self.game_over = Some(GameOverStatus::LockOut);
            events.push(EngineEvent::GameOver {
                reason: GameOverStatus::LockOut,
            });
            // A dying lock sends nothing: the attack block below is never
            // reached, so the clear neither cancels this player's pending
            // queue nor emits AttackSent — death takes priority over offense,
            // and no consumer has to scan a batch for a trailing GameOver to
            // know whether its AttackSent counts.
            return;
        }

        // Versus: a clear's attack first cancels this player's own pending
        // garbage (oldest first); only the remainder leaves the board.
        if let Some(award) = award {
            let attack = attack_lines(
                award.action,
                award.back_to_back_bonus,
                combo_before,
                self.board.is_empty(),
            );
            let net = self.garbage.cancel(attack);
            if net > 0 {
                events.push(EngineEvent::AttackSent { lines: net });
            }
        }

        // Versus: pending garbage rises after a lock that cleared nothing —
        // between lock and spawn, so a fatal rise is an ordinary block-out for
        // the next spawn (or an outright overflow here).
        if lines_cleared == 0 {
            self.rise_pending_garbage(events);
            if self.game_over.is_some() {
                return;
            }
        }

        self.spawn_next_piece(events);
    }

    /// Apply the batches due after a clear-less lock (see `garbage.rs`): insert
    /// each through the same board primitive the out-of-band seam uses, emit
    /// [`EngineEvent::GarbageInserted`], and latch a block-out if the stack is
    /// forced past the ceiling. Runs between lock and spawn, so there is no
    /// active piece to bury — overflow is the only fatal case here.
    fn rise_pending_garbage(&mut self, events: &mut Vec<EngineEvent>) {
        let rising = self.garbage.rise(self.config.garbage_cap);
        let mut inserted = 0u32;
        let mut overflow = false;
        for batch in rising {
            overflow |= self
                .board
                .insert_garbage_lines(batch.lines as usize, batch.hole_col);
            inserted += batch.lines;
        }
        if inserted > 0 {
            events.push(EngineEvent::GarbageInserted { lines: inserted });
        }
        if overflow {
            self.game_over = Some(GameOverStatus::BlockOut);
            events.push(EngineEvent::GameOver {
                reason: GameOverStatus::BlockOut,
            });
        }
    }

    fn score_lock_result(
        &mut self,
        t_spin: Option<TSpinKind>,
        lines_cleared: usize,
        events: &mut Vec<EngineEvent>,
    ) -> Option<ScoreAward> {
        let action = EngineScoreAction::from_lock_result(t_spin, lines_cleared);
        self.score(action, events)
    }

    /// Score `action`, push the award event, and hand the award back to the
    /// caller (the lock path feeds it to the versus attack table).
    fn score(
        &mut self,
        action: EngineScoreAction,
        events: &mut Vec<EngineEvent>,
    ) -> Option<ScoreAward> {
        let award = score_action(&mut self.score_state, self.config.goal_system, action);
        if let Some(score_award) = award {
            push_score_award(events, score_award);
        }
        award
    }

    fn advance_time(&mut self, dt_seconds: f32, events: &mut Vec<EngineEvent>) {
        if dt_seconds == 0.0 || self.active.is_none() {
            return;
        }

        if self.active.as_ref().is_some_and(ActivePiece::landed) {
            self.advance_lock_timer(dt_seconds, events);
        } else {
            self.advance_gravity(dt_seconds);
        }
    }

    fn advance_lock_timer(&mut self, dt_seconds: f32, events: &mut Vec<EngineEvent>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let remaining = active.lock_timer_seconds() - dt_seconds;
        active.set_lock_timer_seconds(remaining);
        if remaining > 0.0 {
            return;
        }

        let active = self.active.take().expect("active piece exists");
        self.lock_active_piece(active, events);
    }

    fn advance_gravity(&mut self, dt_seconds: f32) {
        self.gravity_accumulator_seconds += dt_seconds;
        let fall_seconds = fall_speed_seconds(self.score_state.level());

        while self.gravity_accumulator_seconds >= fall_seconds {
            self.gravity_accumulator_seconds -= fall_seconds;

            let Some(active) = self.active.as_mut() else {
                return;
            };
            let Some(origin) =
                active
                    .piece()
                    .try_move(&self.board, active.origin(), MoveDirection::Down)
            else {
                update_landing_state(&self.board, &self.config, active, false, false);
                self.gravity_accumulator_seconds = 0.0;
                return;
            };

            active.move_to(origin, crate::engine::PieceAction::Fall);
            update_landing_state(&self.board, &self.config, active, false, false);
            if active.landed() {
                self.gravity_accumulator_seconds = 0.0;
                return;
            }
        }
    }

    fn board_snapshot_cells(&self) -> Vec<SnapshotCell> {
        self.board
            .cells()
            .into_iter()
            .filter_map(|cell| match cell.cell_kind() {
                CellKind::Some(piece_type) => Some(SnapshotCell {
                    x: cell.coords().0,
                    y: cell.coords().1,
                    piece_type,
                    garbage: false,
                }),
                CellKind::Garbage => Some(SnapshotCell {
                    x: cell.coords().0,
                    y: cell.coords().1,
                    piece_type: PieceType::I, // legacy fill colour; see SnapshotCell::piece_type
                    garbage: true,
                }),
                CellKind::None | CellKind::Wall => None,
            })
            .collect()
    }

    fn ghost_snapshot_cells(&self) -> Vec<SnapshotCell> {
        let Some(active) = self.active.as_ref() else {
            return Vec::new();
        };
        let mut origin = active.origin();
        while let Some(next_origin) =
            active
                .piece()
                .try_move(&self.board, origin, MoveDirection::Down)
        {
            origin = next_origin;
        }

        piece_snapshot_cells(active.piece(), origin)
    }
}

fn active_piece_snapshot(active: &ActivePiece, config: &EngineConfig) -> ActivePieceSnapshot {
    let lock_timer_fraction = if active.lock_timer_active() {
        (active.lock_timer_seconds() / config.lock_down_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ActivePieceSnapshot {
        piece_type: active.piece_type(),
        rotation: active.rotation(),
        origin: active.origin(),
        cells: piece_snapshot_cells(active.piece(), active.origin()),
        hold_used: active.hold_used_on_this_piece(),
        landed: active.landed(),
        lock_timer_seconds: active.lock_timer_seconds(),
        lock_timer_fraction,
    }
}

fn piece_snapshot_cells(piece: &Piece, origin: (isize, isize)) -> Vec<SnapshotCell> {
    piece
        .cells()
        .into_iter()
        .map(|(x, y)| SnapshotCell {
            x: x + origin.0,
            y: y + origin.1,
            piece_type: piece.piece_type(),
            garbage: false,
        })
        .collect()
}

fn active_is_grounded(board: &Board, active: &ActivePiece) -> bool {
    active
        .piece()
        .try_move(board, active.origin(), MoveDirection::Down)
        .is_none()
}

fn update_landing_state(
    board: &Board,
    config: &EngineConfig,
    active: &mut ActivePiece,
    was_landed: bool,
    grounded_move_or_rotation: bool,
) {
    if !active_is_grounded(board, active) {
        active.mark_airborne();
        return;
    }

    if !was_landed {
        active.mark_landed();
        active.reset_lock_timer(config.lock_down_seconds);
    } else if grounded_move_or_rotation {
        apply_grounded_move_or_rotation(active, config.lock_down_mode, config.lock_down_seconds);
    }
}

fn push_score_award(events: &mut Vec<EngineEvent>, score_award: ScoreAward) {
    events.push(EngineEvent::ScoreAwarded {
        action: score_award.action,
        score: score_award.score,
        total_score: score_award.total_score,
        back_to_back_bonus: score_award.back_to_back_bonus,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::lock_down::LOCK_DOWN_SECONDS;

    fn active_piece_type(engine: &Engine) -> PieceType {
        engine.snapshot().active.expect("active piece").piece_type
    }

    fn active_origin(engine: &Engine) -> (isize, isize) {
        engine.snapshot().active.expect("active piece").origin
    }

    fn lock_piece(engine: &mut Engine, active: ActivePiece) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        engine.lock_active_piece(active, &mut events);
        events
    }

    fn sorted_cell_coords(cells: &[SnapshotCell]) -> Vec<(isize, isize)> {
        let mut coords = cells
            .iter()
            .map(|cell| (cell.x, cell.y))
            .collect::<Vec<_>>();
        coords.sort();
        coords
    }

    #[test]
    fn new_engine_has_deterministic_preview_queue() {
        let config = EngineConfig::default();
        let left = Engine::new(config.clone(), 42);
        let right = Engine::new(config, 42);

        assert_eq!(left.snapshot(), right.snapshot());
        assert_eq!(left.snapshot().next_queue.len(), 5);
        assert!(left.snapshot().active.is_none());
    }

    #[test]
    fn zero_delta_step_spawns_first_piece_with_immediate_drop() {
        let config = EngineConfig::default();
        let mut engine = Engine::new(config.clone(), 0);
        let first_piece_type = engine.snapshot().next_queue[0];
        let piece = Piece::from(first_piece_type);
        let board =
            Board::with_top_margin(config.board_width, config.visible_height, BUFFER_HEIGHT);
        let spawn_origin = piece.spawn_coords(config.board_width, config.visible_height);
        let expected_origin = piece
            .try_move(&board, spawn_origin, MoveDirection::Down)
            .unwrap_or(spawn_origin);

        // The spawn is snapshot state, not an event: active goes None -> Some
        // across a step that emits nothing else.
        assert!(engine.snapshot().active.is_none());
        assert!(engine.step(InputFrame::default()).is_empty());

        let snapshot = engine.snapshot();
        let active = snapshot.active.expect("spawned active piece");
        assert_eq!(active.piece_type, first_piece_type);
        assert_eq!(active.origin, expected_origin);
        assert_eq!(active.cells.len(), 4);
        assert!(snapshot.board_cells.is_empty());
    }

    #[test]
    fn spawn_block_out_ends_game_before_immediate_drop() {
        let config = EngineConfig::default();
        let mut engine = Engine::new(config.clone(), 0);
        let first_piece_type = engine.snapshot().next_queue[0];
        let piece = Piece::from(first_piece_type);
        let spawn_origin = piece.spawn_coords(config.board_width, config.visible_height);
        let blocking_cell = piece.cells()[0];
        assert!(engine.board.set(
            spawn_origin.0 + blocking_cell.0,
            spawn_origin.1 + blocking_cell.1,
            CellKind::Some(PieceType::O),
        ));

        assert_eq!(
            engine.step(InputFrame::default()),
            vec![EngineEvent::GameOver {
                reason: GameOverStatus::BlockOut
            }]
        );
        assert_eq!(engine.snapshot().game_over, Some(GameOverStatus::BlockOut));
        assert!(engine.snapshot().active.is_none());
    }

    #[test]
    fn hold_without_existing_hold_stores_active_and_spawns_next_piece() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        let initial_queue = engine.snapshot().next_queue;
        let first_piece_type = initial_queue[0];
        let second_piece_type = initial_queue[1];
        engine.step(InputFrame::default());

        assert_eq!(
            engine.step(InputFrame {
                hold: true,
                ..InputFrame::default()
            }),
            vec![EngineEvent::Held {
                held: first_piece_type,
                active: second_piece_type,
            }]
        );

        let snapshot = engine.snapshot();
        let active = snapshot.active.expect("held active piece");
        assert_eq!(snapshot.hold, Some(first_piece_type));
        assert_eq!(active.piece_type, second_piece_type);
        assert_eq!(active.rotation, PieceRotation::R0);
        assert!(active.hold_used);
    }

    #[test]
    fn hold_can_only_be_used_once_per_active_piece() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        engine.step(InputFrame {
            hold: true,
            ..InputFrame::default()
        });
        let before = engine.snapshot();

        assert!(engine
            .step(InputFrame {
                hold: true,
                ..InputFrame::default()
            })
            .is_empty());
        assert_eq!(engine.snapshot(), before);
    }

    #[test]
    fn hold_with_existing_piece_swaps_to_north_facing_spawn() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let outgoing = active_piece_type(&engine);
        let held = if outgoing == PieceType::I {
            PieceType::T
        } else {
            PieceType::I
        };
        engine.hold = Some(held);

        engine.step(InputFrame {
            hold: true,
            ..InputFrame::default()
        });

        let snapshot = engine.snapshot();
        let active = snapshot.active.expect("swapped active piece");
        assert_eq!(snapshot.hold, Some(outgoing));
        assert_eq!(active.piece_type, held);
        assert_eq!(active.rotation, PieceRotation::R0);
        assert!(active.hold_used);
    }

    #[test]
    fn resolved_horizontal_input_moves_active_piece_once() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let before = engine.snapshot().active.expect("active piece");
        let expected_origin = (before.origin.0 - 1, before.origin.1);

        // Movement is snapshot state: the step emits nothing, and the origin
        // shifts by exactly one cell.
        assert!(engine
            .step(InputFrame {
                left: true,
                ..InputFrame::default()
            })
            .is_empty());

        assert_eq!(
            engine.snapshot().active.expect("moved active piece").origin,
            expected_origin
        );
    }

    #[test]
    fn blocked_horizontal_input_does_not_move_or_emit_event() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let active = engine.snapshot().active.expect("active piece");
        let blocking_cell = active.cells[0];
        assert!(engine.board.set(
            blocking_cell.x - 1,
            blocking_cell.y,
            CellKind::Some(PieceType::O),
        ));

        assert!(engine
            .step(InputFrame {
                left: true,
                ..InputFrame::default()
            })
            .is_empty());

        assert_eq!(
            engine
                .snapshot()
                .active
                .expect("blocked active piece")
                .origin,
            active.origin
        );
    }

    #[test]
    fn resolved_soft_drop_moves_active_piece_down_once() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let before = engine.snapshot().active.expect("active piece");
        let expected_origin = (before.origin.0, before.origin.1 - 1);

        assert_eq!(
            engine.step(InputFrame {
                soft_drop: true,
                ..InputFrame::default()
            }),
            vec![EngineEvent::ScoreAwarded {
                action: EngineScoreAction::SoftDrop,
                score: 1,
                total_score: 1,
                back_to_back_bonus: false,
            }]
        );
        assert_eq!(active_origin(&engine), expected_origin);
    }

    #[test]
    fn resolved_rotation_uses_srs_kicks() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        let origin = (3, 18);
        let piece = Piece::from(PieceType::T);
        let (rotation, kicked_origin, kick_number) = piece
            .try_rotate_with_kicks(&engine.board, origin, PieceRotation::R90)
            .expect("T should rotate on an empty board");
        engine.active = Some(ActivePiece::new(PieceType::T, origin));

        assert_eq!(
            engine.step(InputFrame {
                rotate_clockwise: true,
                ..InputFrame::default()
            }),
            vec![EngineEvent::Rotated {
                piece_type: PieceType::T,
                rotation,
                origin: kicked_origin,
                kick_number,
            }]
        );

        let active = engine.snapshot().active.expect("rotated active piece");
        assert_eq!(active.rotation, PieceRotation::R90);
        assert_eq!(active.origin, kicked_origin);
    }

    #[test]
    fn engine_sets_point_5_override_when_kick_5_rotates_into_a_t_slot() {
        // Regression for the §7.5 point-5 override: `rotate_active_piece` must
        // *compute* whether SRS test 5 placed the T into a T-slot, not hardcode it
        // false. Build a board where a T's clockwise rotation can only resolve via
        // test 5 (tests 1-4 all collide) and lands in a 3-corner T-slot, then drive
        // the rotation through the real `step` path.
        //
        // T at R0 origin (4,5); after the test-5 kick (-1,-2) it rests at R90
        // origin (3,3), center (4,4). Blockers at (5,5)+(4,7) fail tests 1-4, and
        // corners (3,5),(5,5),(3,3) make the landing a T-slot. The test-5 landing
        // cells (4,5),(4,4),(5,4),(4,3) are left clear so the kick succeeds.
        let mut engine = Engine::new(EngineConfig::default(), 0);
        for (x, y) in [(5, 5), (4, 7), (3, 5), (3, 3)] {
            engine.set_cell(x, y, CellKind::Some(PieceType::O));
        }
        engine.set_active(ActivePiece::new(PieceType::T, (4, 5)));

        let events = engine.step(InputFrame {
            rotate_clockwise: true,
            ..InputFrame::default()
        });

        // The rotation resolved via SRS test 5 (kick number 5)...
        assert!(
            events.iter().any(|event| matches!(
                event,
                EngineEvent::Rotated {
                    kick_number: 5,
                    rotation: PieceRotation::R90,
                    ..
                }
            )),
            "expected a kick-5 R90 rotation, got {events:?}",
        );
        // ...and the engine recorded the kick-5-into-T-slot override, so the spin
        // classifies Full and survives a later soft-drop / lateral tap (§12.4).
        // Before the fix this flag was never set through `step` (only by tests
        // hand-building an `ActivePiece`), so a real TST silently lost its scoring.
        assert!(
            engine
                .active
                .as_ref()
                .expect("active piece after rotation")
                .used_kick_5_into_t_slot(),
            "kick-5 rotation into a T-slot must set the sticky point-5 override",
        );
    }

    #[test]
    fn hard_drop_locks_piece_to_board_and_spawns_next_piece() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        let initial_queue = engine.snapshot().next_queue;
        let first_piece_type = initial_queue[0];
        let second_piece_type = initial_queue[1];
        engine.step(InputFrame::default());

        let events = engine.step(InputFrame {
            hard_drop: true,
            ..InputFrame::default()
        });

        assert!(matches!(
            events.as_slice(),
            [
                EngineEvent::HardDropped {
                    piece_type,
                    cells_dropped,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::HardDrop { cells },
                    score,
                    total_score,
                    back_to_back_bonus: false,
                },
                EngineEvent::Locked {
                    piece_type: locked_piece_type,
                    lines_cleared: 0,
                },
            ] if *piece_type == first_piece_type
                && *locked_piece_type == first_piece_type
                && *cells_dropped > 0
                && *cells == *cells_dropped
                && *score == *cells_dropped * 2
                && *total_score == *score
        ));

        // The same step spawned the next piece: visible in the snapshot.
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.board_cells.len(), 4);
        assert_eq!(
            snapshot.active.expect("next active piece").piece_type,
            second_piece_type
        );
    }

    #[test]
    fn gravity_uses_accumulated_delta_time_to_fall_one_row() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let before = engine.snapshot().active.expect("active piece");
        let half_fall = fall_speed_seconds(engine.snapshot().level) / 2.0;

        assert!(engine
            .step(InputFrame {
                dt_seconds: half_fall,
                ..InputFrame::default()
            })
            .is_empty());
        assert_eq!(
            engine.snapshot().active.expect("active piece").origin,
            before.origin
        );

        // The second half-interval crosses the fall threshold: the piece falls
        // exactly one row (gravity is snapshot state, the step emits nothing).
        assert!(engine
            .step(InputFrame {
                dt_seconds: half_fall,
                ..InputFrame::default()
            })
            .is_empty());
        assert_eq!(
            active_origin(&engine),
            (before.origin.0, before.origin.1 - 1)
        );
    }

    #[test]
    fn gravity_landing_starts_lock_timer_before_locking_on_next_frame() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.active = Some(ActivePiece::new(PieceType::T, (3, 0)));

        // One full fall interval drops the T one row onto the floor.
        assert!(engine
            .step(InputFrame {
                dt_seconds: fall_speed_seconds(engine.snapshot().level),
                ..InputFrame::default()
            })
            .is_empty());
        assert_eq!(active_origin(&engine), (3, -1));

        let active = engine.snapshot().active.expect("landed active piece");
        assert!(active.landed);
        assert_eq!(active.lock_timer_seconds, LOCK_DOWN_SECONDS);
        assert!(engine.snapshot().board_cells.is_empty());

        let events = engine.step(InputFrame {
            dt_seconds: LOCK_DOWN_SECONDS,
            ..InputFrame::default()
        });

        assert!(matches!(
            events.as_slice(),
            [EngineEvent::Locked {
                piece_type: PieceType::T,
                lines_cleared: 0,
            }]
        ));
        assert_eq!(engine.snapshot().board_cells.len(), 4);
        // The locking step also spawned the next piece.
        assert!(engine.snapshot().active.is_some());
    }

    #[test]
    fn extended_lock_down_budget_stops_resetting_after_fifteen_grounded_moves() {
        let config = EngineConfig {
            board_width: 40,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 0);
        let mut active = ActivePiece::new(PieceType::T, (20, -1));
        active.mark_landed();
        active.reset_lock_timer(LOCK_DOWN_SECONDS);
        engine.active = Some(active);

        for _ in 0..crate::engine::EXTENDED_LOCK_RESET_BUDGET {
            // Each grounded left move succeeds: origin.x shifts by exactly -1.
            let x_before = active_origin(&engine).0;
            assert!(engine
                .step(InputFrame {
                    left: true,
                    ..InputFrame::default()
                })
                .is_empty());
            assert_eq!(active_origin(&engine).0, x_before - 1);
            assert_eq!(
                engine
                    .active
                    .as_ref()
                    .expect("active piece")
                    .lock_timer_seconds(),
                LOCK_DOWN_SECONDS
            );
        }

        engine
            .active
            .as_mut()
            .expect("active piece")
            .set_lock_timer_seconds(0.1);
        assert_eq!(
            engine
                .active
                .as_ref()
                .expect("active piece")
                .grounded_move_rotate_count_since_lowest(),
            crate::engine::EXTENDED_LOCK_RESET_BUDGET
        );

        // The 16th grounded move still succeeds physically (origin.x shifts)
        // but no longer resets the drained timer.
        let x_before = active_origin(&engine).0;
        assert!(engine
            .step(InputFrame {
                left: true,
                ..InputFrame::default()
            })
            .is_empty());
        assert_eq!(active_origin(&engine).0, x_before - 1);
        assert_eq!(
            engine
                .active
                .as_ref()
                .expect("active piece")
                .lock_timer_seconds(),
            0.1
        );

        let events = engine.step(InputFrame {
            dt_seconds: 0.1,
            ..InputFrame::default()
        });
        assert!(matches!(
            events.as_slice(),
            [EngineEvent::Locked {
                piece_type: PieceType::T,
                lines_cleared: 0,
            }]
        ));
        // The expiring lock spawned the next piece in the same step.
        assert!(engine.snapshot().active.is_some());
    }

    #[test]
    fn lock_line_clear_scores_single_and_advances_fixed_goal() {
        let config = EngineConfig {
            board_width: 4,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 0);
        let active = ActivePiece::new(PieceType::I, (0, -2));

        let events = lock_piece(&mut engine, active);

        assert!(matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::I,
                    lines_cleared: 1,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::Single,
                    score: 100,
                    total_score: 100,
                    back_to_back_bonus: false,
                },
                // Clearing the 4-wide board empties it: a perfect clear, whose
                // 10 attack lines leave uncancelled (nothing is pending).
                EngineEvent::AttackSent { lines: 10 },
            ]
        ));

        let snapshot = engine.snapshot();
        assert!(
            snapshot.active.is_some(),
            "the lock spawned the next piece in the same step"
        );
        assert_eq!(snapshot.score, 100);
        assert_eq!(snapshot.lines, 1);
        assert_eq!(snapshot.goal_remaining, 9);
        assert!(!snapshot.back_to_back_active);
    }

    #[test]
    fn lock_tetris_scores_back_to_back_bonus_on_second_qualifying_clear() {
        fn fill_tetris_well(engine: &mut Engine) {
            for y in 0..4 {
                for x in 0..3 {
                    assert!(engine.board.set(x, y, CellKind::Some(PieceType::O)));
                }
            }
        }

        fn vertical_i() -> ActivePiece {
            let mut active = ActivePiece::new(PieceType::I, (1, 0));
            active.rotate_to(
                PieceRotation::R90,
                (1, 0),
                RotationDirection::Clockwise,
                1,
                false,
            );
            active
        }

        let config = EngineConfig {
            board_width: 4,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 0);

        fill_tetris_well(&mut engine);
        let first_events = lock_piece(&mut engine, vertical_i());
        assert!(matches!(
            first_events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::I,
                    lines_cleared: 4,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::Tetris,
                    score: 800,
                    total_score: 800,
                    back_to_back_bonus: false,
                },
                // Tetris (4) + perfect clear (10) on the emptied 4-wide board.
                EngineEvent::AttackSent { lines: 14 },
            ]
        ));
        assert!(engine.snapshot().active.is_some(), "lock spawns the next");
        assert!(engine.snapshot().back_to_back_active);

        fill_tetris_well(&mut engine);
        let second_events = lock_piece(&mut engine, vertical_i());
        assert!(matches!(
            second_events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::I,
                    lines_cleared: 4,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::Tetris,
                    score: 1200,
                    total_score: 2000,
                    back_to_back_bonus: true,
                },
                // Tetris (4) + B2B (1) + combo index 1 (0) + perfect clear (10).
                EngineEvent::AttackSent { lines: 15 },
            ]
        ));
        assert!(engine.snapshot().active.is_some(), "lock spawns the next");
        assert_eq!(engine.snapshot().score, 2000);
    }

    #[test]
    fn lock_uses_t_spin_classifier_for_score_action() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        for (x, y) in [(4, 6), (6, 6), (4, 4)] {
            assert!(engine.board.set(x, y, CellKind::Some(PieceType::O)));
        }
        let mut active = ActivePiece::new(PieceType::T, (4, 4));
        active.rotate_to(
            PieceRotation::R0,
            (4, 4),
            RotationDirection::Clockwise,
            1,
            false,
        );

        let events = lock_piece(&mut engine, active);

        assert!(matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 0,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::TSpin {
                        kind: TSpinKind::Full,
                        lines: 0,
                    },
                    score: 400,
                    total_score: 400,
                    back_to_back_bonus: false,
                },
            ]
        ));
        assert!(engine.snapshot().active.is_some(), "lock spawns the next");
        assert_eq!(engine.snapshot().score, 400);
    }

    #[test]
    fn snapshot_ghost_cells_match_hard_drop_landing_cells() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let ghost_cells = sorted_cell_coords(&engine.snapshot().ghost_cells);

        engine.step(InputFrame {
            hard_drop: true,
            ..InputFrame::default()
        });

        assert_eq!(
            sorted_cell_coords(&engine.snapshot().board_cells),
            ghost_cells
        );
    }

    #[test]
    fn snapshot_ghost_cells_follow_horizontal_movement() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.step(InputFrame::default());
        let before = sorted_cell_coords(&engine.snapshot().ghost_cells);

        engine.step(InputFrame {
            left: true,
            ..InputFrame::default()
        });

        let after = sorted_cell_coords(&engine.snapshot().ghost_cells);
        assert_eq!(
            after,
            before
                .into_iter()
                .map(|(x, y)| (x - 1, y))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn garbage_burying_the_active_piece_tops_out_like_a_spawn_block_out() {
        let mut engine = Engine::new(EngineConfig::default(), 42);
        engine.step(InputFrame::default()); // spawn the first piece
        assert!(engine.snapshot().active.is_some());

        // Enough garbage to bury any spawn pose (the hole sits at column 0, away
        // from the spawn columns).
        let topped = engine.insert_garbage(25, 0);

        assert!(topped);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.game_over, Some(GameOverStatus::BlockOut));
        assert!(
            snapshot.active.is_none(),
            "the buried piece is dropped, matching the spawn-collision end state"
        );
        // The latched game-over makes the next step a no-op: no respawn happens.
        engine.step(InputFrame::default());
        assert!(engine.snapshot().active.is_none());
    }

    #[test]
    fn snapshot_bag_remainder_matches_the_deal_stream_truth() {
        // The exported remainder must equal "all seven minus what the current bag
        // has dealt", where the current bag is the one containing the next piece
        // beyond the revealed queue. A same-seed generator replays the engine's
        // deal stream as ground truth; piece `i` of the stream belongs to bag
        // `i / 7`. Exercised across several bag boundaries and a hold (which
        // consumes one extra deal when the hold slot is empty).
        for seed in [0u64, 7, 42] {
            let stream: Vec<PieceType> = crate::engine::PieceGenerator::with_seed(seed)
                .take(64)
                .collect();
            // Tall field so naive center hard-drops never top out; bag accounting
            // is board-independent.
            let config = EngineConfig {
                visible_height: 40,
                ..EngineConfig::default()
            };
            let mut engine = Engine::new(config, seed);
            engine.step(InputFrame::default()); // spawn piece 0
            let mut consumed = 6usize; // 1 active + the 5-piece preview

            for k in 0..20 {
                if k == 3 {
                    // An empty-hold swap pulls one extra piece from the generator.
                    engine.step(InputFrame {
                        hold: true,
                        ..InputFrame::default()
                    });
                    consumed += 1;
                }

                let bag_start = (consumed / 7) * 7;
                let dealt_this_bag = &stream[bag_start..consumed];
                let mut expected: Vec<PieceType> = PieceType::all()
                    .into_iter()
                    .filter(|pt| !dealt_this_bag.contains(pt))
                    .collect();
                if consumed.is_multiple_of(7) {
                    expected.clear(); // a bag boundary exports an empty remainder
                }

                let mut remainder = engine.snapshot().bag_remainder;
                remainder.sort_by_key(|pt| *pt as u8);
                expected.sort_by_key(|pt| *pt as u8);
                assert_eq!(
                    remainder, expected,
                    "seed {seed}: bag remainder diverged from the deal stream at piece {k}"
                );

                engine.step(InputFrame {
                    hard_drop: true,
                    ..InputFrame::default()
                });
                consumed += 1;
            }
        }
    }

    // ---- Versus garbage rules (queue / cancellation / rising) ----

    /// Fill row `y` except the columns in `skip` (a line-clear precondition).
    fn fill_row_except(engine: &mut Engine, y: isize, skip: &[isize]) {
        for x in 0..10 {
            if !skip.contains(&x) {
                engine.set_cell(x, y, CellKind::Some(PieceType::J));
            }
        }
    }

    #[test]
    fn queued_garbage_is_pending_until_a_clear_less_lock() {
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.queue_garbage(3);
        assert_eq!(engine.snapshot().pending_garbage_total(), 3);
        assert!(
            engine.snapshot().board_cells.is_empty(),
            "queueing alone must not touch the board"
        );

        // A lock that clears nothing: the pending garbage rises beneath it.
        let events = lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 10)));
        assert!(events.contains(&EngineEvent::GarbageInserted { lines: 3 }));
        assert_eq!(engine.snapshot().pending_garbage_total(), 0);
        // 3 garbage rows of 9 cells (one hole each) plus the locked O.
        let cells = engine.snapshot().board_cells;
        assert_eq!(cells.len(), 3 * 9 + 4);
        // The snapshot tells attack from stack: risen rows (y < 3) carry the
        // garbage flag, the player's own locked piece does not.
        for cell in &cells {
            assert_eq!(
                cell.garbage,
                cell.y < 3,
                "garbage flag wrong at ({}, {})",
                cell.x,
                cell.y
            );
        }
    }

    #[test]
    fn a_clear_defers_rising_and_attack_cancels_pending_first() {
        let mut engine = Engine::new(EngineConfig::default(), 7);
        // Rows 0-1 ready for a Double at cols 4-5; a stray high cell prevents a
        // perfect clear from inflating the attack.
        fill_row_except(&mut engine, 0, &[4, 5]);
        fill_row_except(&mut engine, 1, &[4, 5]);
        engine.set_cell(0, 5, CellKind::Some(PieceType::J));
        engine.queue_garbage(2);

        let events = lock_piece(&mut engine, ActivePiece::new(PieceType::O, (3, -1)));
        // The clear defers rising entirely...
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EngineEvent::GarbageInserted { .. })),
            "a clearing lock must not let garbage rise"
        );
        // ...and the Double's 1 attack line is absorbed by the oldest pending
        // line instead of leaving the board.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EngineEvent::AttackSent { .. })),
            "a fully-cancelled attack sends nothing"
        );
        assert_eq!(engine.snapshot().pending_garbage_total(), 1);
    }

    #[test]
    fn net_attack_is_what_survives_cancellation() {
        let mut engine = Engine::new(EngineConfig::default(), 7);
        // Rows 0-1 filled except cols 4-5 and nothing else: the Double perfect-
        // clears, so the gross attack is 1 (Double) + 10 (perfect clear) = 11.
        fill_row_except(&mut engine, 0, &[4, 5]);
        fill_row_except(&mut engine, 1, &[4, 5]);
        engine.queue_garbage(1);

        let events = lock_piece(&mut engine, ActivePiece::new(PieceType::O, (3, -1)));
        assert!(
            events.contains(&EngineEvent::AttackSent { lines: 10 }),
            "11 gross attack minus 1 cancelled pending line leaves 10: {events:?}"
        );
        assert_eq!(engine.snapshot().pending_garbage_total(), 0);
    }

    #[test]
    fn rising_respects_the_garbage_cap_and_a_split_batch_keeps_its_hole() {
        let config = EngineConfig {
            garbage_cap: 4,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 7);
        engine.queue_garbage(6); // one 6-line batch: a single hole column

        lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 12)));
        assert_eq!(
            engine.snapshot().pending_garbage_total(),
            2,
            "the cap holds 2 of the 6 lines back"
        );
        lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 14)));
        assert_eq!(engine.snapshot().pending_garbage_total(), 0);

        // All 6 garbage rows came from one batch, so they share one hole column.
        let cells = engine.snapshot().board_cells;
        for y in 0..6isize {
            let filled: Vec<isize> = cells.iter().filter(|c| c.y == y).map(|c| c.x).collect();
            assert_eq!(filled.len(), 9, "garbage row {y} has exactly one hole");
            let hole: Vec<isize> = (0..10).filter(|x| !filled.contains(x)).collect();
            let row0_hole: Vec<isize> = {
                let f: Vec<isize> = cells.iter().filter(|c| c.y == 0).map(|c| c.x).collect();
                (0..10).filter(|x| !f.contains(x)).collect()
            };
            assert_eq!(hole, row0_hole, "split halves of one batch share the hole");
        }
    }

    #[test]
    fn an_overflowing_rise_is_a_block_out() {
        let config = EngineConfig {
            garbage_cap: 64,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 7);
        engine.queue_garbage(64); // taller than the whole 40-row backing

        let events = lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 10)));
        assert!(events.contains(&EngineEvent::GameOver {
            reason: GameOverStatus::BlockOut
        }));
        assert_eq!(
            engine.snapshot().game_over,
            Some(GameOverStatus::BlockOut),
            "an overflowing rise ends the game in-band"
        );
    }

    #[test]
    fn a_dying_lock_sends_no_attack_and_leaves_pending_intact() {
        // A piece can lock entirely above the skyline yet still clear buffer
        // rows (full-matrix clears are deliberate). Death takes priority over
        // offense: the lock-out lock emits NO AttackSent — its clear neither
        // cancels this player's pending queue nor reaches the opponent — so no
        // consumer has to scan an event batch for a trailing GameOver to know
        // whether an attack counts.
        let mut engine = Engine::new(EngineConfig::default(), 7);
        // Rows 30-31 (buffer zone; visible height is 20) filled except cols
        // 4-5: the O completes both, a Double that would perfect-clear (gross
        // attack 11) — while locking entirely above the skyline.
        for y in [30, 31] {
            fill_row_except(&mut engine, y, &[4, 5]);
        }
        engine.queue_garbage(1);

        let events = lock_piece(&mut engine, ActivePiece::new(PieceType::O, (3, 29)));
        assert!(events.contains(&EngineEvent::GameOver {
            reason: GameOverStatus::LockOut
        }));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, EngineEvent::AttackSent { .. })),
            "a dying lock must not send: {events:?}"
        );
        assert_eq!(
            engine.snapshot().pending_garbage_total(),
            1,
            "a dying lock must not consume its own pending queue either"
        );
    }

    #[test]
    fn a_finished_engine_accepts_no_garbage() {
        // End a game by an overflowing rise (everything in one lock).
        let mut engine = Engine::new(
            EngineConfig {
                garbage_cap: 64,
                ..EngineConfig::default()
            },
            7,
        );
        engine.queue_garbage(64); // taller than the 40-row backing
        lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 10)));
        assert_eq!(engine.snapshot().game_over, Some(GameOverStatus::BlockOut));

        // Post-mortem, both out-of-band seams are inert.
        let cells_before = engine.snapshot().board_cells.len();
        engine.queue_garbage(5);
        assert_eq!(
            engine.snapshot().pending_garbage_total(),
            0,
            "a finished game accepts no more attack"
        );
        assert!(
            engine.insert_garbage(3, 0),
            "inserting into a finished game reports topped-out"
        );
        assert_eq!(
            engine.snapshot().board_cells.len(),
            cells_before,
            "the final board is a faithful record"
        );
        assert_eq!(
            engine.snapshot().game_over,
            Some(GameOverStatus::BlockOut),
            "the latched reason is never rewritten post-mortem"
        );
    }

    #[test]
    fn garbage_holes_are_engine_seed_deterministic() {
        let board_after = |seed: u64| {
            let mut engine = Engine::new(EngineConfig::default(), seed);
            engine.queue_garbage(2);
            engine.queue_garbage(3);
            lock_piece(&mut engine, ActivePiece::new(PieceType::O, (4, 10)));
            sorted_cell_coords(&engine.snapshot().board_cells)
        };
        assert_eq!(board_after(42), board_after(42), "same seed, same holes");
    }
}
