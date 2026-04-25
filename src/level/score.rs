use crate::core::{MoveDirection, PieceType};
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

#[derive(Resource, Default)]
pub struct Scorer {
    action_history: Vec<ActionEvent>,
    pub score: usize,
    pub lines: usize,
    pub combo: usize,
    pub difficult_action: bool,
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

        self.action_history.push(action);
    }

    pub fn lock_piece(&mut self, lines: usize) -> Vec<ScoreType> {
        if lines > 4 {
            warn!("ignoring invalid line clear count: {lines}");
            return vec![];
        }

        self.lines += lines;

        let last_action = match self.action_history.last() {
            Some(ActionEvent::HardDrop(_)) => self
                .action_history
                .len()
                .checked_sub(2)
                .and_then(|idx| self.action_history.get(idx)),
            _ => self.action_history.last(),
        };

        let mut difficult_action = false;
        let (mut score, spin_score_types) = match (lines, last_action) {
            (1, Some(ActionEvent::Rotation(PieceType::T, 3, _))) => {
                difficult_action = true;
                (800, vec![ScoreType::TSpin, ScoreType::Single])
            }
            (3, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, _))) => {
                difficult_action = true;
                (1600, vec![ScoreType::TSpin, ScoreType::Triple])
            }
            (2, Some(ActionEvent::Rotation(PieceType::T, 3 | 4, _))) => {
                difficult_action = true;
                (1200, vec![ScoreType::TSpin, ScoreType::Double])
            }
            (2, Some(ActionEvent::Rotation(PieceType::T, _, true))) => {
                difficult_action = true;
                (400, vec![ScoreType::MiniTSpin, ScoreType::Double])
            }
            (1, Some(ActionEvent::Rotation(PieceType::T, _, true))) => {
                difficult_action = true;
                (200, vec![ScoreType::MiniTSpin, ScoreType::Single])
            }
            (0, Some(ActionEvent::Rotation(PieceType::T, _, true))) => {
                (100, vec![ScoreType::MiniTSpin])
            }
            (0, Some(ActionEvent::Rotation(PieceType::T, 3, _))) => (400, vec![ScoreType::TSpin]),
            _ => (0, vec![]),
        };

        let mut score_types = spin_score_types;

        match lines {
            0 => {
                self.combo = 0;
            }
            1..=4 => {
                if score_types.is_empty() {
                    let (line_clear_score, score_type) = match lines {
                        1 => (100, ScoreType::Single),
                        2 => (300, ScoreType::Double),
                        3 => (500, ScoreType::Triple),
                        4 => {
                            difficult_action = true;
                            (800, ScoreType::Tetris)
                        }
                        _ => unreachable!("line count is constrained by the match arm"),
                    };
                    score += line_clear_score;
                    score_types.push(score_type);
                }

                if self.combo > 0 {
                    self.score += self.combo * 50;
                    score_types.push(ScoreType::Combo(self.combo + 1));
                }

                self.combo += 1;

                if difficult_action && self.difficult_action {
                    score = score * 3 / 2;
                    score_types.push(ScoreType::BackToBack);
                }

                self.difficult_action = difficult_action;
            }
            _ => unreachable!("line count is validated before scoring"),
        }

        self.score += score;
        score_types
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
| Combo | 50 xx combo count xx level |
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
    Combo(usize),
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

        info!("score types: {:?}", score_types);
        info!("action history: {:?}", scorer.action_history);
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
        assert_eq!(scorer.combo, 1);
        assert_eq!(score_types, vec![ScoreType::Double]);
    }

    #[test]
    fn consecutive_line_clears_add_combo_bonus() {
        let mut scorer = Scorer::default();

        assert_eq!(scorer.lock_piece(1), vec![ScoreType::Single]);
        assert_eq!(
            scorer.lock_piece(2),
            vec![ScoreType::Double, ScoreType::Combo(2)]
        );

        assert_eq!(scorer.score, 450);
        assert_eq!(scorer.combo, 2);
    }

    #[test]
    fn no_line_clear_resets_combo() {
        let mut scorer = Scorer::default();

        scorer.lock_piece(1);
        scorer.lock_piece(0);

        assert_eq!(scorer.combo, 0);
    }

    #[test]
    fn t_spin_double_scores_from_last_rotation_before_hard_drop() {
        let mut scorer = Scorer::default();
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 3, false));
        scorer.record_action(ActionEvent::HardDrop(1));

        let score_types = scorer.lock_piece(2);

        assert_eq!(scorer.score, 1202);
        assert_eq!(score_types, vec![ScoreType::TSpin, ScoreType::Double]);
        assert!(scorer.difficult_action);
    }

    #[test]
    fn back_to_back_applies_to_current_difficult_clear() {
        let mut scorer = Scorer::default();

        assert_eq!(scorer.lock_piece(4), vec![ScoreType::Tetris]);
        assert_eq!(
            scorer.lock_piece(4),
            vec![
                ScoreType::Tetris,
                ScoreType::Combo(2),
                ScoreType::BackToBack
            ]
        );

        assert_eq!(scorer.score, 2050);
    }

    #[test]
    fn mini_t_spin_no_line_scores_without_combo() {
        let mut scorer = Scorer::default();
        scorer.record_action(ActionEvent::Rotation(PieceType::T, 2, true));

        let score_types = scorer.lock_piece(0);

        assert_eq!(scorer.score, 100);
        assert_eq!(scorer.combo, 0);
        assert_eq!(score_types, vec![ScoreType::MiniTSpin]);
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
