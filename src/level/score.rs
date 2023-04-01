use crate::core::{MoveDirection, PieceType};
use crate::level::common::{ActionEvent, LevelState, PlacingEvent};
use bevy::prelude::*;

pub struct ScorePlugin;

impl Plugin for ScorePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<Vec<ScoreType>>()
            .insert_resource(Scorer::default()).add_systems(
            (append_action_history, update_score)
                .chain()
                .in_set(OnUpdate(LevelState::Playing)),
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

fn append_action_history(mut ev_action: EventReader<ActionEvent>,
                         mut scorer: ResMut<Scorer>
) {
    for ev in ev_action.iter() {
        scorer.action_history.push(ev.clone());

        // hard/soft drop action scores
        match ev {
            ActionEvent::HardDrop(lines) => {
                scorer.score += 2 * lines;
            }
            ActionEvent::Movement(MoveDirection::Down) => {
                scorer.score += 1;
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
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

fn update_score(mut ev_placing: EventReader<PlacingEvent>,
                mut ev_score_types: EventWriter<Vec<ScoreType>>,
                mut scorer: ResMut<Scorer>) {
    for ev in ev_placing.iter() {

        if matches!(ev, PlacingEvent::Placed) { // we don't want to update score on Placed event
            continue;
        }
        // add lines cleared
        if let PlacingEvent::Locked(lines) = ev {
            scorer.lines += lines;
        }
        let mut score_types = vec![];

        // find last action which is not a hard drop
        let last_action = match scorer.action_history.last() {
            Some(ActionEvent::HardDrop(_)) => {
                if scorer.action_history.len() < 2 {
                    None
                } else {
                    scorer.action_history.get(scorer.action_history.len() - 2)
                }
            }
            _ => scorer.action_history.last(),
        };

        let mut difficult_action = false;

        let (mut score, spin_score_types) = match (ev, last_action) {
            // T-Spin Single (T piece, 1 line cleared, 3 corners touching)
            (PlacingEvent::Locked(1), Some(ActionEvent::Rotation(_, PieceType::T, 3, _))) => {
                difficult_action = true;
                (800, vec![ScoreType::TSpin, ScoreType::Single])
            }

            // T-Spin Triple (T piece, 3 lines cleared, 3+ corners touching)
            (PlacingEvent::Locked(3), Some(ActionEvent::Rotation(_, PieceType::T, 3, _)))
            | (PlacingEvent::Locked(3), Some(ActionEvent::Rotation(_, PieceType::T, 4, _))) => {
                difficult_action = true;
                (1600, vec![ScoreType::TSpin, ScoreType::Triple])
            }

            // T-Spin Double (T piece, 2 lines cleared, 3+ corners touching)
            (PlacingEvent::Locked(2), Some(ActionEvent::Rotation(_, PieceType::T, 3, _)))
            | (PlacingEvent::Locked(2), Some(ActionEvent::Rotation(_, PieceType::T, 4, _))) => {
                difficult_action = true;
                (1200, vec![ScoreType::TSpin, ScoreType::Double])
            }

            // Mini T-Spin Double (wall kicked, T piece, 2 lines cleared)
            (PlacingEvent::Locked(2), Some(ActionEvent::Rotation(_, PieceType::T, _, true))) => {
                difficult_action = true;
                (400, vec![ScoreType::MiniTSpin, ScoreType::Double])
            }

            // Mini T-Spin Single (wall kicked, T piece, 1 line cleared)
            (PlacingEvent::Locked(1), Some(ActionEvent::Rotation(_, PieceType::T, _, true))) => {
                difficult_action = true;
                (200, vec![ScoreType::MiniTSpin, ScoreType::Single])
            }

            // Mini T-Spin no line (wall kicked, T piece, 0 lines cleared)
            (PlacingEvent::Locked(0), Some(ActionEvent::Rotation(_, PieceType::T, _, true))) => {
                (100, vec![ScoreType::MiniTSpin])
            }

            // T-Spin no line(s) (T piece, 0 lines cleared, 3 corners touching)
            (PlacingEvent::Locked(0), Some(ActionEvent::Rotation(_, PieceType::T, 3, _))) => {
                (400, vec![ScoreType::TSpin])
            }

            _ => (0, vec![]),
        };

        score_types.extend(spin_score_types);

        // update scores from line clears
        match ev {
            PlacingEvent::Locked(0) => {
                scorer.combo = 0; // clear combo if no lines are cleared
            }
            PlacingEvent::Locked(lines) => {
                // if currently no spin is active, add score for line clears
                if score_types.is_empty() {
                    let (line_clear_score, score_type) = match lines {
                        1 => (100, ScoreType::Single),
                        2 => (300, ScoreType::Double),
                        3 => (500, ScoreType::Triple),
                        4 => {
                            difficult_action = true;
                            (800, ScoreType::Tetris)
                        }
                        _ => unreachable!("lines cleared must be between 0 and 4"),
                    };
                    score += line_clear_score;
                    score_types.push(score_type);
                }

                if scorer.combo > 0 {
                    scorer.score += scorer.combo * 50; // each combo adds 50 points
                    score_types.push(ScoreType::Combo(scorer.combo + 1)); // add combo to score types
                }

                scorer.combo += 1; // increment combo

                if difficult_action && scorer.difficult_action {
                    score = (scorer.score as f32 * 1.5) as usize; // back-to-back difficult line clears
                    score_types.push(ScoreType::BackToBack);
                }

                scorer.difficult_action = difficult_action;
            }

            _ => {}
        }

        scorer.score += score;

        info!("score types: {:?}", score_types);
        info!("action history: {:?}", scorer.action_history);
        if !score_types.is_empty() {
            ev_score_types.send(score_types);
        }
    }
}
