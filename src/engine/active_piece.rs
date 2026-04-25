use crate::engine::pieces::{Piece, PieceRotation, PieceType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PieceAction {
    Spawn,
    Fall,
    Move,
    Rotate,
    SoftDrop,
    HardDrop,
    HoldSwap,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RotationDirection {
    Clockwise,
    Counterclockwise,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActivePiece {
    piece: Piece,
    origin: (isize, isize),
    lock_timer_seconds: f32,
    lock_timer_active: bool,
    landed: bool,
    lowest_y_reached: isize,
    grounded_move_rotate_count_since_lowest: u8,
    hold_used_on_this_piece: bool,
    last_successful_action: PieceAction,
    last_rotation_direction: Option<RotationDirection>,
    last_rotation_kick_number: Option<u8>,
    used_kick_5_into_t_slot: bool,
}

impl ActivePiece {
    pub fn new(piece_type: PieceType, origin: (isize, isize)) -> Self {
        Self {
            piece: Piece::from(piece_type),
            origin,
            lock_timer_seconds: 0.0,
            lock_timer_active: false,
            landed: false,
            lowest_y_reached: origin.1,
            grounded_move_rotate_count_since_lowest: 0,
            hold_used_on_this_piece: false,
            last_successful_action: PieceAction::Spawn,
            last_rotation_direction: None,
            last_rotation_kick_number: None,
            used_kick_5_into_t_slot: false,
        }
    }

    pub fn piece(&self) -> &Piece {
        &self.piece
    }

    pub fn piece_type(&self) -> PieceType {
        self.piece.piece_type()
    }

    pub fn rotation(&self) -> PieceRotation {
        self.piece.rotation()
    }

    pub fn origin(&self) -> (isize, isize) {
        self.origin
    }

    pub fn lock_timer_seconds(&self) -> f32 {
        self.lock_timer_seconds
    }

    pub fn lock_timer_active(&self) -> bool {
        self.lock_timer_active
    }

    pub fn landed(&self) -> bool {
        self.landed
    }

    pub fn lowest_y_reached(&self) -> isize {
        self.lowest_y_reached
    }

    pub fn grounded_move_rotate_count_since_lowest(&self) -> u8 {
        self.grounded_move_rotate_count_since_lowest
    }

    pub fn hold_used_on_this_piece(&self) -> bool {
        self.hold_used_on_this_piece
    }

    pub fn last_successful_action(&self) -> PieceAction {
        self.last_successful_action
    }

    pub fn last_rotation_direction(&self) -> Option<RotationDirection> {
        self.last_rotation_direction
    }

    pub fn last_rotation_kick_number(&self) -> Option<u8> {
        self.last_rotation_kick_number
    }

    pub fn used_kick_5_into_t_slot(&self) -> bool {
        self.used_kick_5_into_t_slot
    }

    pub fn move_to(&mut self, origin: (isize, isize), action: PieceAction) {
        assert!(
            matches!(
                action,
                PieceAction::Fall
                    | PieceAction::Move
                    | PieceAction::SoftDrop
                    | PieceAction::HardDrop
            ),
            "move_to only accepts movement/drop actions"
        );

        self.origin = origin;
        self.last_successful_action = action;
        self.last_rotation_direction = None;
        self.last_rotation_kick_number = None;
        self.update_lowest_y(origin.1);
    }

    pub fn rotate_to(
        &mut self,
        rotation: PieceRotation,
        origin: (isize, isize),
        direction: RotationDirection,
        kick_number: u8,
        entered_t_slot_with_kick_5: bool,
    ) {
        assert!(
            (1..=5).contains(&kick_number),
            "SRS kick number must be in 1..=5"
        );

        self.piece.rotate_to(rotation);
        self.origin = origin;
        self.last_successful_action = PieceAction::Rotate;
        self.last_rotation_direction = Some(direction);
        self.last_rotation_kick_number = Some(kick_number);
        if kick_number == 5 && entered_t_slot_with_kick_5 {
            self.used_kick_5_into_t_slot = true;
        }
        self.update_lowest_y(origin.1);
    }

    pub fn mark_landed(&mut self) {
        self.landed = true;
        self.lock_timer_active = true;
    }

    pub fn mark_airborne(&mut self) {
        self.landed = false;
        self.lock_timer_active = false;
    }

    pub fn set_lock_timer_seconds(&mut self, seconds: f32) {
        self.lock_timer_seconds = seconds.max(0.0);
    }

    pub fn reset_lock_timer(&mut self, seconds: f32) {
        self.lock_timer_seconds = seconds.max(0.0);
        self.lock_timer_active = true;
    }

    pub fn record_grounded_move_or_rotate(&mut self) {
        self.grounded_move_rotate_count_since_lowest = self
            .grounded_move_rotate_count_since_lowest
            .saturating_add(1);
    }

    pub fn mark_hold_used(&mut self) {
        self.hold_used_on_this_piece = true;
        self.last_successful_action = PieceAction::HoldSwap;
        self.last_rotation_direction = None;
        self.last_rotation_kick_number = None;
    }

    fn update_lowest_y(&mut self, y: isize) {
        if y < self.lowest_y_reached {
            self.lowest_y_reached = y;
            self.grounded_move_rotate_count_since_lowest = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_active_piece_initializes_spawn_state() {
        let active = ActivePiece::new(PieceType::T, (3, 19));

        assert_eq!(active.piece_type(), PieceType::T);
        assert_eq!(active.rotation(), PieceRotation::R0);
        assert_eq!(active.origin(), (3, 19));
        assert_eq!(active.lowest_y_reached(), 19);
        assert_eq!(active.last_successful_action(), PieceAction::Spawn);
        assert_eq!(active.last_rotation_direction(), None);
        assert_eq!(active.last_rotation_kick_number(), None);
        assert!(!active.lock_timer_active());
        assert!(!active.landed());
        assert!(!active.hold_used_on_this_piece());
        assert!(!active.used_kick_5_into_t_slot());
    }

    #[test]
    fn falling_below_previous_lowest_resets_grounded_budget() {
        let mut active = ActivePiece::new(PieceType::T, (3, 19));
        active.record_grounded_move_or_rotate();
        active.record_grounded_move_or_rotate();

        active.move_to((3, 18), PieceAction::Fall);

        assert_eq!(active.lowest_y_reached(), 18);
        assert_eq!(active.grounded_move_rotate_count_since_lowest(), 0);
        assert_eq!(active.last_successful_action(), PieceAction::Fall);
    }

    #[test]
    fn grounded_move_budget_saturates() {
        let mut active = ActivePiece::new(PieceType::T, (3, 19));

        for _ in 0..=u8::MAX {
            active.record_grounded_move_or_rotate();
        }

        assert_eq!(active.grounded_move_rotate_count_since_lowest(), u8::MAX);
    }

    #[test]
    fn rotation_records_direction_kick_and_point_5_override() {
        let mut active = ActivePiece::new(PieceType::T, (3, 19));

        active.rotate_to(
            PieceRotation::R90,
            (4, 19),
            RotationDirection::Clockwise,
            5,
            true,
        );

        assert_eq!(active.rotation(), PieceRotation::R90);
        assert_eq!(active.origin(), (4, 19));
        assert_eq!(active.last_successful_action(), PieceAction::Rotate);
        assert_eq!(
            active.last_rotation_direction(),
            Some(RotationDirection::Clockwise)
        );
        assert_eq!(active.last_rotation_kick_number(), Some(5));
        assert!(active.used_kick_5_into_t_slot());
    }

    #[test]
    fn lock_timer_and_landed_state_are_explicit() {
        let mut active = ActivePiece::new(PieceType::O, (3, 19));

        active.mark_landed();
        active.reset_lock_timer(0.5);
        assert!(active.landed());
        assert!(active.lock_timer_active());
        assert_eq!(active.lock_timer_seconds(), 0.5);

        active.mark_airborne();
        assert!(!active.landed());
        assert!(!active.lock_timer_active());

        active.set_lock_timer_seconds(-1.0);
        assert_eq!(active.lock_timer_seconds(), 0.0);
    }

    #[test]
    fn hold_used_is_tracked_per_piece() {
        let mut active = ActivePiece::new(PieceType::I, (3, 18));

        active.mark_hold_used();

        assert!(active.hold_used_on_this_piece());
        assert_eq!(active.last_successful_action(), PieceAction::HoldSwap);
    }
}
