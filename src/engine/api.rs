use crate::engine::active_piece::ActivePiece;
use crate::engine::board::{Board, CellKind};
use crate::engine::game_over::{is_block_out, is_lock_out};
use crate::engine::generator::PieceGenerator;
use crate::engine::goals::GoalSystem;
use crate::engine::gravity::{fall_speed_seconds, MIN_LEVEL};
use crate::engine::lock_clear::lock_and_clear;
use crate::engine::lock_down::{apply_grounded_move_or_rotation, LockDownMode, LOCK_DOWN_SECONDS};
use crate::engine::pieces::{MoveDirection, Piece, PieceRotation, PieceType};
use crate::engine::scoring::{score_action, EngineScoreAction, ScoreAward, ScoreState};
use crate::engine::t_spin::{classify_t_spin, TSpinKind};
use crate::engine::RotationDirection;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub board_width: usize,
    pub visible_height: usize,
    pub buffer_height: usize,
    pub preview_count: usize,
    pub lock_down_mode: LockDownMode,
    pub lock_down_seconds: f32,
    pub starting_level: u8,
    pub goal_system: GoalSystem,
    pub das_delay_seconds: f32,
    pub das_repeat_seconds: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            board_width: 10,
            visible_height: 20,
            buffer_height: 20,
            preview_count: 5,
            lock_down_mode: LockDownMode::Extended,
            lock_down_seconds: LOCK_DOWN_SECONDS,
            starting_level: MIN_LEVEL,
            goal_system: GoalSystem::Fixed,
            das_delay_seconds: 0.167,
            das_repeat_seconds: 0.033,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Spawned {
        piece_type: PieceType,
    },
    Moved {
        piece_type: PieceType,
        direction: MoveDirection,
        origin: (isize, isize),
    },
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
    pub piece_type: PieceType,
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
    pub game_over: Option<GameOverStatus>,
}

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
}

impl Engine {
    pub fn new(config: EngineConfig, seed: u64) -> Self {
        let board = Board::with_top_margin(
            config.board_width,
            config.visible_height,
            config.buffer_height,
        );
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
            game_over: self.game_over,
        }
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
        events.push(EngineEvent::Spawned { piece_type });
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
        }
        events.push(EngineEvent::Moved {
            piece_type: active.piece_type(),
            direction,
            origin,
        });
        if direction == MoveDirection::Down {
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

        active.rotate_to(rotation, origin, direction, kick_number, false);
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
        self.score_lock_result(t_spin, lines_cleared, events);

        if lock_out {
            self.game_over = Some(GameOverStatus::LockOut);
            events.push(EngineEvent::GameOver {
                reason: GameOverStatus::LockOut,
            });
            return;
        }

        self.spawn_next_piece(events);
    }

    fn score_lock_result(
        &mut self,
        t_spin: Option<TSpinKind>,
        lines_cleared: usize,
        events: &mut Vec<EngineEvent>,
    ) {
        let action = EngineScoreAction::from_lock_result(t_spin, lines_cleared);
        self.score(action, events);
    }

    fn score(&mut self, action: EngineScoreAction, events: &mut Vec<EngineEvent>) {
        if let Some(score_award) =
            score_action(&mut self.score_state, self.config.goal_system, action)
        {
            push_score_award(events, score_award);
        }
    }

    fn advance_time(&mut self, dt_seconds: f32, events: &mut Vec<EngineEvent>) {
        if dt_seconds == 0.0 || self.active.is_none() {
            return;
        }

        if self.active.as_ref().is_some_and(ActivePiece::landed) {
            self.advance_lock_timer(dt_seconds, events);
        } else {
            self.advance_gravity(dt_seconds, events);
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

    fn advance_gravity(&mut self, dt_seconds: f32, events: &mut Vec<EngineEvent>) {
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
            events.push(EngineEvent::Moved {
                piece_type: active.piece_type(),
                direction: MoveDirection::Down,
                origin,
            });
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

    fn active_piece_type(engine: &Engine) -> PieceType {
        engine.snapshot().active.expect("active piece").piece_type
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
        let board = Board::with_top_margin(
            config.board_width,
            config.visible_height,
            config.buffer_height,
        );
        let spawn_origin = piece.spawn_coords(config.board_width, config.visible_height);
        let expected_origin = piece
            .try_move(&board, spawn_origin, MoveDirection::Down)
            .unwrap_or(spawn_origin);

        assert_eq!(
            engine.step(InputFrame::default()),
            vec![EngineEvent::Spawned {
                piece_type: first_piece_type
            }]
        );

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
            vec![
                EngineEvent::Spawned {
                    piece_type: second_piece_type,
                },
                EngineEvent::Held {
                    held: first_piece_type,
                    active: second_piece_type,
                },
            ]
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

        assert_eq!(
            engine.step(InputFrame {
                left: true,
                ..InputFrame::default()
            }),
            vec![EngineEvent::Moved {
                piece_type: before.piece_type,
                direction: MoveDirection::Left,
                origin: expected_origin,
            }]
        );

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
            vec![
                EngineEvent::Moved {
                    piece_type: before.piece_type,
                    direction: MoveDirection::Down,
                    origin: expected_origin,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::SoftDrop,
                    score: 1,
                    total_score: 1,
                    back_to_back_bonus: false,
                },
            ]
        );
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
                EngineEvent::Spawned {
                    piece_type: spawned_piece_type,
                },
            ] if *piece_type == first_piece_type
                && *locked_piece_type == first_piece_type
                && *spawned_piece_type == second_piece_type
                && *cells_dropped > 0
                && *cells == *cells_dropped
                && *score == *cells_dropped * 2
                && *total_score == *score
        ));

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

        assert_eq!(
            engine.step(InputFrame {
                dt_seconds: half_fall,
                ..InputFrame::default()
            }),
            vec![EngineEvent::Moved {
                piece_type: before.piece_type,
                direction: MoveDirection::Down,
                origin: (before.origin.0, before.origin.1 - 1),
            }]
        );
    }

    #[test]
    fn gravity_landing_starts_lock_timer_before_locking_on_next_frame() {
        let mut engine = Engine::new(EngineConfig::default(), 0);
        engine.active = Some(ActivePiece::new(PieceType::T, (3, 0)));

        assert_eq!(
            engine.step(InputFrame {
                dt_seconds: fall_speed_seconds(engine.snapshot().level),
                ..InputFrame::default()
            }),
            vec![EngineEvent::Moved {
                piece_type: PieceType::T,
                direction: MoveDirection::Down,
                origin: (3, -1),
            }]
        );

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
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 0,
                },
                EngineEvent::Spawned { .. },
            ]
        ));
        assert_eq!(engine.snapshot().board_cells.len(), 4);
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
            assert_eq!(
                engine
                    .step(InputFrame {
                        left: true,
                        ..InputFrame::default()
                    })
                    .len(),
                1
            );
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

        assert_eq!(
            engine
                .step(InputFrame {
                    left: true,
                    ..InputFrame::default()
                })
                .len(),
            1
        );
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
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 0,
                },
                EngineEvent::Spawned { .. },
            ]
        ));
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
                EngineEvent::Spawned { .. },
            ]
        ));

        let snapshot = engine.snapshot();
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
                EngineEvent::Spawned { .. },
            ]
        ));
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
                EngineEvent::Spawned { .. },
            ]
        ));
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
                EngineEvent::Spawned { .. },
            ]
        ));
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
}
