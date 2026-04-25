use crate::engine::active_piece::ActivePiece;
use crate::engine::board::{Board, CellKind};
use crate::engine::game_over::{is_block_out, is_lock_out};
use crate::engine::generator::PieceGenerator;
use crate::engine::goals::{GoalProgress, GoalSystem};
use crate::engine::gravity::MIN_LEVEL;
use crate::engine::lock_down::{LockDownMode, LOCK_DOWN_SECONDS};
use crate::engine::pieces::{MoveDirection, Piece, PieceRotation, PieceType};
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivePieceSnapshot {
    pub piece_type: PieceType,
    pub rotation: PieceRotation,
    pub origin: (isize, isize),
    pub cells: Vec<SnapshotCell>,
    pub hold_used: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    pub config: EngineConfig,
    pub board_cells: Vec<SnapshotCell>,
    pub active: Option<ActivePieceSnapshot>,
    pub hold: Option<PieceType>,
    pub next_queue: Vec<PieceType>,
    pub score: usize,
    pub lines: usize,
    pub level: u8,
    pub goal_remaining: usize,
    pub game_over: Option<GameOverStatus>,
}

pub struct Engine {
    config: EngineConfig,
    board: Board,
    active: Option<ActivePiece>,
    generator: PieceGenerator,
    next_queue: Vec<PieceType>,
    hold: Option<PieceType>,
    score: usize,
    lines: usize,
    goal_progress: GoalProgress,
    game_over: Option<GameOverStatus>,
}

impl Engine {
    pub fn new(config: EngineConfig, seed: u64) -> Self {
        let board = Board::with_top_margin(
            config.board_width,
            config.visible_height,
            config.buffer_height,
        );
        let goal_progress = GoalProgress::new(config.goal_system, config.starting_level);
        let mut engine = Self {
            config,
            board,
            active: None,
            generator: PieceGenerator::with_seed(seed),
            next_queue: Vec::new(),
            hold: None,
            score: 0,
            lines: 0,
            goal_progress,
            game_over: None,
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

        if input.hold {
            self.hold_active_piece(&mut events);
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

        events
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        EngineSnapshot {
            config: self.config.clone(),
            board_cells: self.board_snapshot_cells(),
            active: self.active.as_ref().map(active_piece_snapshot),
            hold: self.hold,
            next_queue: self.next_queue.clone(),
            score: self.score,
            lines: self.lines,
            level: self.goal_progress.level(),
            goal_remaining: self.goal_progress.remaining(),
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
        self.active = Some(active);
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
        events.push(EngineEvent::Moved {
            piece_type: active.piece_type(),
            direction,
            origin,
        });
    }

    fn rotate_active_piece(&mut self, direction: RotationDirection, events: &mut Vec<EngineEvent>) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
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
        self.lock_active_piece(active, events);
    }

    fn lock_active_piece(&mut self, active: ActivePiece, events: &mut Vec<EngineEvent>) {
        let piece_type = active.piece_type();
        let lock_out = is_lock_out(active.piece(), active.origin(), self.config.visible_height);
        for cell in piece_snapshot_cells(active.piece(), active.origin()) {
            self.board
                .set(cell.x, cell.y, CellKind::Some(cell.piece_type));
        }

        let lines_cleared = self.board.clear_lines();
        self.lines += lines_cleared;
        events.push(EngineEvent::Locked {
            piece_type,
            lines_cleared,
        });

        if lock_out {
            self.game_over = Some(GameOverStatus::LockOut);
            events.push(EngineEvent::GameOver {
                reason: GameOverStatus::LockOut,
            });
            return;
        }

        self.spawn_next_piece(events);
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
}

fn active_piece_snapshot(active: &ActivePiece) -> ActivePieceSnapshot {
    ActivePieceSnapshot {
        piece_type: active.piece_type(),
        rotation: active.rotation(),
        origin: active.origin(),
        cells: piece_snapshot_cells(active.piece(), active.origin()),
        hold_used: active.hold_used_on_this_piece(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn active_piece_type(engine: &Engine) -> PieceType {
        engine.snapshot().active.expect("active piece").piece_type
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
            vec![EngineEvent::Moved {
                piece_type: before.piece_type,
                direction: MoveDirection::Down,
                origin: expected_origin,
            }]
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
        ));

        let snapshot = engine.snapshot();
        assert_eq!(snapshot.board_cells.len(), 4);
        assert_eq!(
            snapshot.active.expect("next active piece").piece_type,
            second_piece_type
        );
    }
}
