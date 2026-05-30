//! Game variants (M1 shared contract): Marathon, Sprint, Ultra.
//!
//! A [`Variant`] selects the *rules around* a single game: which
//! [`EngineConfig`] overrides apply, the goal system, the end condition, the
//! display name, and which high-score category the result is filed under. The
//! three variants are implemented inline here (rather than as separate plugins)
//! because they cross-cut the info-panel (which reads goal/time/score) and
//! high-scores (which file per category).
//!
//! Wiring:
//! * [`ActiveVariant`] resource (default [`Variant::Marathon`]) holds the chosen
//!   variant. Mode-select writes it; the engine bridge reads its
//!   [`VariantDef::apply_engine_overrides`] when building the engine.
//! * [`VariantProgress`] tracks wall-clock elapsed time for the active run.
//! * [`check_variant_end_conditions`] runs each frame while `Playing` and
//!   transitions to [`GameState::GameOver`] when the variant's goal/limit is met
//!   (engine-driven block/lock-out is handled separately by the level plugin).

use bevy::prelude::*;

use crate::engine::{EngineConfig, EngineSnapshot, GoalSystem, MAX_LEVEL};
use crate::level::engine_bridge::LatestSnapshot;
use crate::GameState;

/// Default Sprint line target (clear N lines as fast as possible).
pub const DEFAULT_SPRINT_LINES: usize = 40;
/// Default Ultra time limit in seconds (score as high as possible in 2 minutes).
pub const DEFAULT_ULTRA_SECONDS: f32 = 120.0;
/// Marathon ends when the player completes the final level (engine [`MAX_LEVEL`]).
pub const MARATHON_END_LEVEL: u8 = MAX_LEVEL;

/// The three single-player modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum Variant {
    /// Endless-style climb to the final level; score-primary.
    Marathon,
    /// Clear a fixed number of lines as fast as possible; time-primary.
    Sprint,
    /// Score as high as possible within a fixed time limit; score-primary.
    Ultra,
}

impl Variant {
    /// All variants in mode-select display order.
    pub const ALL: [Variant; 3] = [Variant::Marathon, Variant::Sprint, Variant::Ultra];

    /// The full rules contract for this variant.
    pub fn def(self) -> VariantDef {
        match self {
            Variant::Marathon => VariantDef {
                variant: self,
                display_name: "Marathon",
                goal_system: GoalSystem::Variable,
                end_condition: EndCondition::ReachLevel(MARATHON_END_LEVEL),
                score_kind: ScoreKind::Score,
                line_target: None,
                time_limit_seconds: None,
            },
            Variant::Sprint => VariantDef {
                variant: self,
                display_name: "Sprint",
                goal_system: GoalSystem::Fixed,
                end_condition: EndCondition::ClearLines(DEFAULT_SPRINT_LINES),
                score_kind: ScoreKind::Time,
                line_target: Some(DEFAULT_SPRINT_LINES),
                time_limit_seconds: None,
            },
            Variant::Ultra => VariantDef {
                variant: self,
                display_name: "Ultra",
                goal_system: GoalSystem::Fixed,
                end_condition: EndCondition::TimeLimit(DEFAULT_ULTRA_SECONDS),
                score_kind: ScoreKind::Score,
                line_target: None,
                time_limit_seconds: Some(DEFAULT_ULTRA_SECONDS),
            },
        }
    }

    pub fn display_name(self) -> &'static str {
        self.def().display_name
    }
}

/// What makes a variant end (besides the engine's own block-/lock-out, which is
/// always fatal).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndCondition {
    /// Marathon: the player completed level `N` (snapshot level reaches it).
    ReachLevel(u8),
    /// Sprint: `N` total lines cleared.
    ClearLines(usize),
    /// Ultra: `secs` of play elapsed.
    TimeLimit(f32),
}

/// Which figure is the "primary" result for ranking on the high-score board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreKind {
    /// Lower is better — rank by elapsed time ascending (Sprint).
    Time,
    /// Higher is better — rank by score descending (Marathon, Ultra).
    Score,
}

/// The resolved rules for a variant. Returned by [`Variant::def`]; consumed by
/// the engine bridge, info-panel, and high-scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariantDef {
    pub variant: Variant,
    pub display_name: &'static str,
    pub goal_system: GoalSystem,
    pub end_condition: EndCondition,
    pub score_kind: ScoreKind,
    /// Sprint line target (None for non-line-target variants).
    pub line_target: Option<usize>,
    /// Ultra time limit in seconds (None for untimed variants).
    pub time_limit_seconds: Option<f32>,
}

impl VariantDef {
    /// Apply this variant's overrides onto a base [`EngineConfig`].
    ///
    /// Currently only the goal system differs per variant; board size,
    /// preview count, and lock-down come from `GameSettings`/`LevelConfig`. Kept
    /// as a single seam so future per-variant engine tweaks land here.
    pub fn apply_engine_overrides(&self, config: &mut EngineConfig) {
        config.goal_system = self.goal_system;
    }
}

/// The chosen variant for the next/current game. Default [`Variant::Marathon`].
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Resource)]
pub struct ActiveVariant(pub Variant);

impl Default for ActiveVariant {
    fn default() -> Self {
        Self(Variant::Marathon)
    }
}

/// Wall-clock progress for the active run. `elapsed_seconds` advances each frame
/// while `Playing`; `ended` latches once an end condition fires so we transition
/// to GameOver exactly once. The info-panel reads `elapsed_seconds` for the Ultra
/// countdown / Sprint timer; high-scores read it for the Sprint time result.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct VariantProgress {
    pub elapsed_seconds: f32,
    pub ended: bool,
}

impl VariantProgress {
    /// Whether the variant-level end condition has been met for `snapshot` at the
    /// current elapsed time. Pure so it can be unit-tested without a running app.
    pub fn end_condition_met(
        def: &VariantDef,
        snapshot: &EngineSnapshot,
        elapsed_seconds: f32,
    ) -> bool {
        match def.end_condition {
            EndCondition::ReachLevel(level) => snapshot.level >= level,
            EndCondition::ClearLines(lines) => snapshot.lines >= lines,
            EndCondition::TimeLimit(limit) => elapsed_seconds >= limit,
        }
    }
}

/// Reset [`VariantProgress`] when a game starts. Registered on `OnEnter(Playing)`.
pub fn reset_variant_progress(mut progress: ResMut<VariantProgress>) {
    *progress = VariantProgress::default();
}

/// Advance the run clock and transition to [`GameState::GameOver`] when the
/// active variant's end condition is met. Runs while `Playing`. Engine-driven
/// game-over (block/lock-out) is handled by the level plugin; this only adds the
/// variant goal/time/limit endings.
pub fn check_variant_end_conditions(
    time: Res<Time>,
    active: Res<ActiveVariant>,
    snapshot: Res<LatestSnapshot>,
    mut progress: ResMut<VariantProgress>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if progress.ended {
        return;
    }
    progress.elapsed_seconds += time.delta_secs();

    let def = active.0.def();
    if VariantProgress::end_condition_met(&def, &snapshot.0, progress.elapsed_seconds) {
        progress.ended = true;
        info!(
            "variant end condition met for {}: transitioning to game over",
            def.display_name
        );
        next_state.set(GameState::GameOver);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn snapshot_with(level: u8, lines: usize) -> EngineSnapshot {
        let mut snap = Engine::new(EngineConfig::default(), 0).snapshot();
        snap.level = level;
        snap.lines = lines;
        snap
    }

    #[test]
    fn default_variant_is_marathon() {
        assert_eq!(ActiveVariant::default().0, Variant::Marathon);
    }

    #[test]
    fn marathon_ends_at_final_level() {
        let def = Variant::Marathon.def();
        assert!(!VariantProgress::end_condition_met(
            &def,
            &snapshot_with(14, 0),
            9_999.0
        ));
        assert!(VariantProgress::end_condition_met(
            &def,
            &snapshot_with(MARATHON_END_LEVEL, 0),
            0.0
        ));
    }

    #[test]
    fn sprint_ends_when_line_target_reached() {
        let def = Variant::Sprint.def();
        assert_eq!(def.line_target, Some(DEFAULT_SPRINT_LINES));
        assert_eq!(def.score_kind, ScoreKind::Time);
        assert!(!VariantProgress::end_condition_met(
            &def,
            &snapshot_with(1, 39),
            0.0
        ));
        assert!(VariantProgress::end_condition_met(
            &def,
            &snapshot_with(1, 40),
            0.0
        ));
    }

    #[test]
    fn ultra_ends_at_time_limit() {
        let def = Variant::Ultra.def();
        assert_eq!(def.score_kind, ScoreKind::Score);
        assert!(!VariantProgress::end_condition_met(
            &def,
            &snapshot_with(1, 0),
            119.9
        ));
        assert!(VariantProgress::end_condition_met(
            &def,
            &snapshot_with(1, 0),
            DEFAULT_ULTRA_SECONDS
        ));
    }

    #[test]
    fn engine_overrides_set_goal_system_per_variant() {
        let mut config = EngineConfig::default();
        Variant::Marathon.def().apply_engine_overrides(&mut config);
        assert_eq!(config.goal_system, GoalSystem::Variable);
        Variant::Sprint.def().apply_engine_overrides(&mut config);
        assert_eq!(config.goal_system, GoalSystem::Fixed);
    }
}
