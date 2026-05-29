//! Score state, now a read-only mirror of the engine (P2.2).
//!
//! Before the migration this module *computed* score from a Bevy-side history of
//! `ActionEvent`s. The engine is now authoritative: it emits
//! [`EngineEvent::ScoreAwarded`] with the final per-action score, running total,
//! and back-to-back flag, and exposes `score` / `lines` / `level` on the
//! snapshot. This module just mirrors those into a [`Scorer`] resource for the
//! UI and translates each award into the on-screen [`ScoreType`] labels.

use crate::engine::{EngineScoreAction, TSpinKind};
use crate::level::engine_bridge::{FrameEvents, LatestSnapshot};
use crate::GameState;
use bevy::prelude::*;

pub struct ScorePlugin;

impl Plugin for ScorePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ScoreTypes>()
            .insert_resource(Scorer::default())
            .add_systems(OnEnter(GameState::Playing), reset_score)
            .add_systems(
                Update,
                (mirror_snapshot_score, emit_score_types)
                    .run_if(in_state(GameState::Playing)),
            );
    }
}

/// Read-only mirror of the engine's scalar score fields, for UI consumption.
#[derive(Resource, PartialEq, Eq)]
pub struct Scorer {
    pub level: usize,
    pub score: usize,
    pub lines: usize,
    pub back_to_back_active: bool,
}

impl Default for Scorer {
    fn default() -> Self {
        Self {
            level: 1,
            score: 0,
            lines: 0,
            back_to_back_active: false,
        }
    }
}

fn reset_score(mut scorer: ResMut<Scorer>) {
    *scorer = Scorer::default();
}

/// Copy the snapshot's scalar score fields into [`Scorer`], only writing on a
/// real change so the UI's `resource_changed::<Scorer>` run-condition fires
/// exactly when the displayed numbers change.
fn mirror_snapshot_score(snapshot: Res<LatestSnapshot>, mut scorer: ResMut<Scorer>) {
    let next = Scorer {
        level: snapshot.0.level as usize,
        score: snapshot.0.score,
        lines: snapshot.0.lines,
        back_to_back_active: snapshot.0.back_to_back_active,
    };
    scorer.set_if_neq(next);
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

/// Translate an engine score action (+ its back-to-back flag) into the labels
/// shown in the score-type popup. Drop actions (soft/hard) and no-clear locks
/// produce no labels.
pub(crate) fn score_types_for(action: EngineScoreAction, back_to_back_bonus: bool) -> Vec<ScoreType> {
    let mut types = match action {
        EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. }
        | EngineScoreAction::NoClear => vec![],
        EngineScoreAction::Single => vec![ScoreType::Single],
        EngineScoreAction::Double => vec![ScoreType::Double],
        EngineScoreAction::Triple => vec![ScoreType::Triple],
        EngineScoreAction::Tetris => vec![ScoreType::Tetris],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines: 0,
        } => vec![ScoreType::MiniTSpin],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines: 1,
        } => vec![ScoreType::MiniTSpin, ScoreType::Single],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 0,
        } => vec![ScoreType::TSpin],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 1,
        } => vec![ScoreType::TSpin, ScoreType::Single],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 2,
        } => vec![ScoreType::TSpin, ScoreType::Double],
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 3,
        } => vec![ScoreType::TSpin, ScoreType::Triple],
        EngineScoreAction::TSpin { .. } => vec![],
    };

    if back_to_back_bonus && !types.is_empty() {
        types.push(ScoreType::BackToBack);
    }
    types
}

/// Single canonical consumer of [`EngineEvent::ScoreAwarded`]: turn each award
/// into a [`ScoreTypes`] message for the popup UI.
fn emit_score_types(frame_events: Res<FrameEvents>, mut ev_score_types: MessageWriter<ScoreTypes>) {
    use crate::engine::EngineEvent;
    for event in &frame_events.0 {
        if let EngineEvent::ScoreAwarded {
            action,
            back_to_back_bonus,
            ..
        } = event
        {
            let types = score_types_for(*action, *back_to_back_bonus);
            if !types.is_empty() {
                ev_score_types.write(ScoreTypes(types));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_clear_labels_single() {
        assert_eq!(
            score_types_for(EngineScoreAction::Single, false),
            vec![ScoreType::Single]
        );
    }

    #[test]
    fn drops_and_no_clear_produce_no_labels() {
        assert!(score_types_for(EngineScoreAction::SoftDrop, false).is_empty());
        assert!(score_types_for(EngineScoreAction::HardDrop { cells: 5 }, false).is_empty());
        assert!(score_types_for(EngineScoreAction::NoClear, false).is_empty());
    }

    #[test]
    fn back_to_back_bonus_appends_label_for_qualifying_clear() {
        assert_eq!(
            score_types_for(EngineScoreAction::Tetris, true),
            vec![ScoreType::Tetris, ScoreType::BackToBack]
        );
    }

    #[test]
    fn t_spin_double_labels_tspin_then_double() {
        assert_eq!(
            score_types_for(
                EngineScoreAction::TSpin {
                    kind: TSpinKind::Full,
                    lines: 2
                },
                false
            ),
            vec![ScoreType::TSpin, ScoreType::Double]
        );
    }

    #[test]
    fn zero_line_t_spin_does_not_append_back_to_back_when_no_bonus() {
        // A zero-line full T-spin scores but starts (does not extend) B2B, so the
        // award carries no back_to_back_bonus and no BackToBack label.
        assert_eq!(
            score_types_for(
                EngineScoreAction::TSpin {
                    kind: TSpinKind::Full,
                    lines: 0
                },
                false
            ),
            vec![ScoreType::TSpin]
        );
    }
}
