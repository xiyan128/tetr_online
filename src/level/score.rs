use crate::engine::{
    breaks_back_to_back, qualifies_for_back_to_back, MoveDirection, PieceType, TSpinKind,
};
use crate::level::common::{ActionEvent, PlacingEvent};
use crate::GameState;
use bevy::prelude::*;

pub struct ScorePlugin;

impl Plugin for ScorePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ScoreTypes>()
            .insert_resource(Scorer::default())
            .add_systems(OnEnter(GameState::InGame), reset_score)
            .add_systems(
                Update,
                (append_action_history, update_score)
                    .chain()
                    .run_if(in_state(GameState::InGame)),
            );
    }
}

#[derive(Resource)]
pub struct Scorer {
    last_lock_action: Option<ActionEvent>,
    pub level: usize,
    pub score: usize,
    pub lines: usize,
    pub back_to_back_active: bool,
}

impl Default for Scorer {
    fn default() -> Self {
        Self {
            last_lock_action: None,
            level: 1,
            score: 0,
            lines: 0,
            back_to_back_active: false,
        }
    }
}

impl Scorer {
    pub fn record_action(&mut self, action: ActionEvent) {
        match action {
            ActionEvent::HardDrop(lines) => {
                self.score += 2 * lines;
            }
            ActionEvent::Movement(MoveDirection::Down) => {
                self.score += 1;
            }
            _ => {}
        }

        if !matches!(action, ActionEvent::HardDrop(_)) {
            self.last_lock_action = Some(action);
        }
    }

    pub fn lock_piece(&mut self, lines: usize) -> Vec<ScoreType> {
        if lines > 4 {
            warn!("ignoring invalid line clear count: {lines}");
            return vec![];
        }

        self.lines += lines;

        let action = score_action(lines, self.last_lock_action.as_ref());
        let (base_score, mut score_types) = action.base_score_and_types(self.level);
        let mut score = base_score;

        if action.qualifies_for_back_to_back() {
            if self.back_to_back_active {
                score += base_score / 2;
                score_types.push(ScoreType::BackToBack);
            } else {
                self.back_to_back_active = true;
            }
        } else if action.breaks_back_to_back() {
            self.back_to_back_active = false;
        }

        self.score += score;
        self.last_lock_action = None;
        score_types
    }
}

fn used_wall_kick(kick_number: u8) -> bool {
    kick_number > 1
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ScoreAction {
    NoClear,
    Single,
    Double,
    Triple,
    Tetris,
    TSpin { kind: TSpinKind, lines: usize },
}

impl ScoreAction {
    fn base_score_and_types(self, level: usize) -> (usize, Vec<ScoreType>) {
        let (base, score_types) = match self {
            ScoreAction::NoClear => (0, vec![]),
            ScoreAction::Single => (100, vec![ScoreType::Single]),
            ScoreAction::Double => (300, vec![ScoreType::Double]),
            ScoreAction::Triple => (500, vec![ScoreType::Triple]),
            ScoreAction::Tetris => (800, vec![ScoreType::Tetris]),
            ScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines: 0,
            } => (100, vec![ScoreType::MiniTSpin]),
            ScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines: 1,
            } => (200, vec![ScoreType::MiniTSpin, ScoreType::Single]),
            ScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 0,
            } => (400, vec![ScoreType::TSpin]),
            ScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 1,
            } => (800, vec![ScoreType::TSpin, ScoreType::Single]),
            ScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            } => (1200, vec![ScoreType::TSpin, ScoreType::Double]),
            ScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 3,
            } => (1600, vec![ScoreType::TSpin, ScoreType::Triple]),
            ScoreAction::TSpin { kind, lines } => {
                warn!("ignoring invalid T-Spin score action: {kind:?} with {lines} lines");
                (0, vec![])
            }
        };

        (base * level, score_types)
    }

    fn qualifies_for_back_to_back(self) -> bool {
        let (t_spin, lines) = self.spin_and_lines();
        qualifies_for_back_to_back(t_spin, lines)
    }

    fn breaks_back_to_back(self) -> bool {
        let (t_spin, lines) = self.spin_and_lines();
        breaks_back_to_back(t_spin, lines)
    }

    fn spin_and_lines(self) -> (Option<TSpinKind>, usize) {
        match self {
            ScoreAction::NoClear => (None, 0),
            ScoreAction::Single => (None, 1),
            ScoreAction::Double => (None, 2),
            ScoreAction::Triple => (None, 3),
            ScoreAction::Tetris => (None, 4),
            ScoreAction::TSpin { kind, lines } => (Some(kind), lines),
        }
    }
}

fn score_action(lines: usize, last_action: Option<&ActionEvent>) -> ScoreAction {
    if let Some(kind) = legacy_t_spin_kind(lines, last_action) {
        return ScoreAction::TSpin { kind, lines };
    }

    match lines {
        0 => ScoreAction::NoClear,
        1 => ScoreAction::Single,
        2 => ScoreAction::Double,
        3 => ScoreAction::Triple,
        4 => ScoreAction::Tetris,
        _ => unreachable!("line count is validated before score_action"),
    }
}

fn legacy_t_spin_kind(lines: usize, last_action: Option<&ActionEvent>) -> Option<TSpinKind> {
    match (lines, last_action) {
        (_, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, 5))) => Some(TSpinKind::Full),
        (1, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, _))) => Some(TSpinKind::Full),
        (2 | 3, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, _))) => Some(TSpinKind::Full),
        (0, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, _))) => Some(TSpinKind::Full),
        (0..=1, Some(ActionEvent::Rotation(PieceType::T, _, kick_number)))
            if used_wall_kick(*kick_number) =>
        {
            Some(TSpinKind::Mini)
        }
        _ => None,
    }
}

fn reset_score(mut scorer: ResMut<Scorer>) {
    *scorer = Scorer::default();
}

/*
| Action |  |
| :--- | :--- |
| Single | 100 xx level |
| Double | 300 xx level |
| Triple | 500 xx level |
| Tetris | 800 xx level; difficult |
| Mini T-Spin no line(s) | 100 xx level |
| T-Spin no line(s) | 400 xx level |
| Mini T-Spin Single | 200 xx level; difficult |
| T-Spin Single | 800 xx level; difficult |
| Mini T-Spin Double (if present) | 400 xx level; difficult |
| T-Spin Double | 1200 xx level; difficult |
| T-Spin Triple | 1600 xx level; difficult |
| Back-to-Back difficult line clears | Action score xx1.5 (excluding soft drop and hard drop) |
| Soft drop | 1 per cell |
| Hard drop | 2 per cell |
 */

fn append_action_history(mut ev_action: MessageReader<ActionEvent>, mut scorer: ResMut<Scorer>) {
    for ev in ev_action.read() {
        scorer.record_action(ev.clone());
    }
}

#[derive(Message, Debug)]
pub struct ScoreTypes(pub Vec<ScoreType>);

#[derive(Debug, PartialEq, Eq)]
pub enum ScoreType {
    Single,
    Double,
    Triple,
    Tetris,
    TSpin,
    MiniTSpin,
    BackToBack,
}

fn update_score(
    mut ev_placing: MessageReader<PlacingEvent>,
    mut ev_score_types: MessageWriter<ScoreTypes>,
    mut scorer: ResMut<Scorer>,
) {
    for ev in ev_placing.read() {
        let score_types = match ev {
            PlacingEvent::Locked(lines) => scorer.lock_piece(*lines),
        };

        if !score_types.is_empty() {
            ev_score_types.write(ScoreTypes(score_types));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_action_scores_manual_drops() {
        let mut scorer = Scorer::default();

        scorer.record_action(ActionEvent::Movement(MoveDirection::Down));
        scorer.record_action(ActionEvent::HardDrop(3));

        assert_eq!(scorer.score, 7);
    }

    #[test]
    fn line_clear_scores_and_tracks_lines() {
        let mut scorer = Scorer::default();

        let score_types = scorer.lock_piece(2);

        assert_eq!(scorer.score, 300);
        assert_eq!(scorer.lines, 2);
        assert_eq!(score_types, vec![ScoreType::Double]);
    }

    #[test]
    fn normal_single_double_and_triple_break_back_to_back() {
        let mut scorer = Scorer::default();

        assert_eq!(scorer.lock_piece(4), vec![ScoreType::Tetris]);
        assert!(scorer.back_to_back_active);

        assert_eq!(scorer.lock_piece(1), vec![ScoreType::Single]);

        assert_eq!(scorer.score, 900);
        assert!(!scorer.back_to_back_active);
    }

    #[test]
    fn t_spin_double_scores_from_last_rotation_before_hard_drop() {
        let mut scorer = Scorer::default();
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        scorer.record_action(ActionEvent::HardDrop(1));

        let score_types = scorer.lock_piece(2);

        assert_eq!(scorer.score, 1202);
        assert_eq!(score_types, vec![ScoreType::TSpin, ScoreType::Double]);
        assert!(scorer.back_to_back_active);
    }

    #[test]
    fn back_to_back_applies_to_subsequent_qualifying_clears() {
        let mut scorer = Scorer::default();

        assert_eq!(scorer.lock_piece(4), vec![ScoreType::Tetris]);
        assert_eq!(
            scorer.lock_piece(4),
            vec![ScoreType::Tetris, ScoreType::BackToBack]
        );

        assert_eq!(scorer.score, 2000);
    }

    #[test]
    fn zero_line_t_spin_scores_without_starting_back_to_back() {
        let mut scorer = Scorer::default();
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));

        let score_types = scorer.lock_piece(0);

        assert_eq!(scorer.score, 400);
        assert!(!scorer.back_to_back_active);
        assert_eq!(score_types, vec![ScoreType::TSpin]);
    }

    #[test]
    fn previous_piece_rotation_does_not_leak_into_next_lock() {
        let mut scorer = Scorer::default();

        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        assert_eq!(
            scorer.lock_piece(2),
            vec![ScoreType::TSpin, ScoreType::Double]
        );

        assert_eq!(scorer.lock_piece(2), vec![ScoreType::Double]);
        assert_eq!(scorer.score, 1500);
        assert!(!scorer.back_to_back_active);
    }

    #[test]
    fn two_line_wall_kick_rotation_scores_as_double_not_mini_t_spin_double() {
        let mut scorer = Scorer::default();

        scorer.record_action(ActionEvent::Rotation(PieceType::T, 2, 2));
        let score_types = scorer.lock_piece(2);

        assert_eq!(score_types, vec![ScoreType::Double]);
        assert_eq!(scorer.score, 300);
        assert!(!scorer.back_to_back_active);
    }

    #[test]
    fn zero_line_t_spin_preserves_existing_back_to_back() {
        let mut scorer = Scorer::default();

        scorer.lock_piece(4);
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        assert_eq!(scorer.lock_piece(0), vec![ScoreType::TSpin]);

        assert!(scorer.back_to_back_active);
        assert_eq!(
            scorer.lock_piece(4),
            vec![ScoreType::Tetris, ScoreType::BackToBack]
        );
        assert_eq!(scorer.score, 2400);
    }

    #[test]
    fn section_13_back_to_back_example_totals_5400_at_level_one() {
        let mut scorer = Scorer::default();

        assert_eq!(scorer.lock_piece(4), vec![ScoreType::Tetris]);

        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        assert_eq!(
            scorer.lock_piece(2),
            vec![ScoreType::TSpin, ScoreType::Double, ScoreType::BackToBack]
        );

        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        assert_eq!(scorer.lock_piece(0), vec![ScoreType::TSpin]);

        assert_eq!(
            scorer.lock_piece(4),
            vec![ScoreType::Tetris, ScoreType::BackToBack]
        );

        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        assert_eq!(
            scorer.lock_piece(1),
            vec![ScoreType::TSpin, ScoreType::Single, ScoreType::BackToBack]
        );

        assert_eq!(scorer.score, 5400);
    }

    #[test]
    fn level_multiplies_line_clear_and_spin_scores_but_not_drop_scores() {
        let mut scorer = Scorer {
            level: 2,
            ..Default::default()
        };

        scorer.record_action(ActionEvent::Movement(MoveDirection::Down));
        scorer.record_action(ActionEvent::HardDrop(2));
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, 1));
        let score_types = scorer.lock_piece(1);

        assert_eq!(score_types, vec![ScoreType::TSpin, ScoreType::Single]);
        assert_eq!(scorer.score, 1605);
    }

    #[test]
    fn score_systems_process_messages_in_order() {
        let mut app = App::new();
        app.add_message::<ActionEvent>()
            .add_message::<PlacingEvent>()
            .add_message::<ScoreTypes>()
            .insert_resource(Scorer::default())
            .add_systems(Update, (append_action_history, update_score).chain());

        app.world_mut()
            .write_message(ActionEvent::Movement(MoveDirection::Down));
        app.world_mut().write_message(PlacingEvent::Locked(1));
        app.update();

        let scorer = app.world().resource::<Scorer>();
        assert_eq!(scorer.score, 101);
        assert_eq!(scorer.lines, 1);
    }
}
