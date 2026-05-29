//! Info-panel feature (A1.3).
//!
//! An in-game side panel shown while
//! [`GameState::Playing`](crate::GameState::Playing) that reflects the active
//! [`Variant`](crate::variant::Variant). It reads
//! [`ActiveVariant`](crate::variant::ActiveVariant),
//! [`VariantProgress`](crate::variant::VariantProgress), the engine
//! [`LatestSnapshot`](crate::level::engine_bridge::LatestSnapshot) and the
//! [`HighScores`](crate::high_scores::HighScores) resource, then renders a fixed
//! ordered set of metric rows beside the matrix:
//!
//! * Mode, Level, Lines, Score, Goal, Time, High Score (+ optional TPM / LPM).
//!
//! Which rows are *meaningful* depends on the variant — Marathon shows
//! level/lines/score & a goal of "level N/MAX"; Sprint shows lines-remaining and
//! the elapsed timer; Ultra shows the time-remaining countdown and score. Rows
//! that don't apply to a variant are hidden rather than removed, so the layout
//! stays stable.
//!
//! Layout: a vertical [`Node`] column pinned to the right edge of the window via
//! absolute positioning, so it sits beside the world-space matrix (which the
//! level renderer centers on its own 2D camera). Reuses
//! [`crate::ui::theme`] + [`crate::ui::widgets`] styling for a consistent look.
//!
//! Touch only this file.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::engine::EngineEvent;
use crate::high_scores::HighScores;
use crate::level::common::LevelSystems;
use crate::level::engine_bridge::{FrameEvents, LatestSnapshot};
use crate::ui::theme;
use crate::variant::{
    ActiveVariant, EndCondition, ScoreKind, Variant, VariantDef, VariantProgress,
    MARATHON_END_LEVEL,
};
use crate::GameState;

/// In-game variant info panel.
pub struct InfoPanelPlugin;

impl Plugin for InfoPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RunStats>()
            // Inspector/scene registration for this feature's resource + markers.
            .register_type::<RunStats>()
            .register_type::<InfoPanelRoot>()
            .register_type::<Metric>()
            .add_systems(
                OnEnter(GameState::Playing),
                (reset_run_stats, spawn_info_panel),
            )
            // Read the same-frame snapshot/events: run after the engine driver
            // has published them. `accumulate_run_stats` consumes this frame's
            // events; `update_info_panel` then paints from snapshot + stats.
            .add_systems(
                Update,
                (accumulate_run_stats, update_info_panel)
                    .chain()
                    .after(LevelSystems::EngineDriver)
                    .run_if(in_state(GameState::Playing))
                    .run_if(resource_exists::<LatestSnapshot>)
                    .run_if(any_with_component::<InfoPanelRoot>),
            );
    }
}

/// Per-run counters the snapshot doesn't expose. Currently just the number of
/// pieces locked (for the optional TPM figure). Reset on entering `Playing`.
#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
struct RunStats {
    pieces_locked: usize,
}

/// Marker for the panel's root node (one per `Playing` session).
#[derive(Component, Reflect)]
#[reflect(Component)]
struct InfoPanelRoot;

/// The figures the panel can display, in render order. Each is a `Text` row
/// tagged with this enum so [`update_info_panel`] can target it.
#[derive(Component, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component)]
enum Metric {
    Mode,
    Level,
    Lines,
    Score,
    Goal,
    Time,
    HighScore,
    Tpm,
    Lpm,
}

impl Metric {
    /// Render order, top to bottom.
    const ORDER: [Metric; 9] = [
        Metric::Mode,
        Metric::Level,
        Metric::Lines,
        Metric::Score,
        Metric::Goal,
        Metric::Time,
        Metric::HighScore,
        Metric::Tpm,
        Metric::Lpm,
    ];
}

fn reset_run_stats(mut stats: ResMut<RunStats>) {
    *stats = RunStats::default();
}

/// Count pieces that locked this frame (drives TPM). Cheap; iterates only the
/// events the engine emitted during this frame's fixed slices.
fn accumulate_run_stats(events: Res<FrameEvents>, mut stats: ResMut<RunStats>) {
    for event in &events.0 {
        if matches!(event, EngineEvent::Locked { .. }) {
            stats.pieces_locked += 1;
        }
    }
}

/// Spawn the right-side metric column. One `Text` per [`Metric`]; values are
/// filled by [`update_info_panel`] on the same/next frame.
fn spawn_info_panel(mut commands: Commands, assets: Res<GameAssets>) {
    let root = commands
        .spawn((
            InfoPanelRoot,
            Node {
                position_type: PositionType::Absolute,
                top: px(16),
                right: px(16),
                width: px(180),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::FlexStart,
                row_gap: px(6),
                padding: UiRect::all(px(12)),
                ..default()
            },
            BackgroundColor(theme::BUTTON_NORMAL.with_alpha(0.85)),
            DespawnOnExit(GameState::Playing),
        ))
        .id();

    for metric in Metric::ORDER {
        let row = commands
            .spawn((
                metric,
                Text::new(""),
                TextFont {
                    font: assets.font.clone(),
                    font_size: theme::LABEL_FONT_SIZE,
                    ..default()
                },
                TextColor(theme::TEXT),
            ))
            .id();
        commands.entity(root).add_child(row);
    }
}

/// Repaint every metric row from the latest snapshot + variant progress, hiding
/// rows that don't apply to the active variant.
fn update_info_panel(
    active: Res<ActiveVariant>,
    snapshot: Res<LatestSnapshot>,
    progress: Res<VariantProgress>,
    high_scores: Res<HighScores>,
    stats: Res<RunStats>,
    mut rows: Query<(&Metric, &mut Text, &mut TextColor, &mut Node)>,
) {
    let variant = active.0;
    let def = variant.def();
    let snap = &snapshot.0;
    let elapsed = progress.elapsed_seconds;
    let best = high_scores.table(variant).first().copied();

    for (metric, mut text, mut color, mut node) in &mut rows {
        let value = metric_value(*metric, &def, snap, elapsed, best, &stats);
        match value {
            Some(line) => {
                text.0 = line;
                node.display = Display::Flex;
                // Mode header reads in the accent color; everything else stays
                // in the standard text color.
                color.0 = if *metric == Metric::Mode {
                    theme::ACCENT
                } else {
                    theme::TEXT
                };
            }
            None => {
                // Not applicable to this variant: collapse the row.
                node.display = Display::None;
            }
        }
    }
}

/// Compute one row's text, or `None` if the metric is hidden for `def`'s
/// variant. Pure-ish (only reads its inputs) so the formatting stays easy to
/// reason about.
fn metric_value(
    metric: Metric,
    def: &VariantDef,
    snap: &crate::engine::EngineSnapshot,
    elapsed: f32,
    best: Option<crate::high_scores::HighScore>,
    stats: &RunStats,
) -> Option<String> {
    match metric {
        Metric::Mode => Some(def.display_name.to_uppercase()),

        // Level is the headline figure for Marathon (climb to MAX); for the
        // fixed-goal variants it's secondary but still informative.
        Metric::Level => Some(match def.variant {
            Variant::Marathon => format!("LEVEL: {}/{}", snap.level, MARATHON_END_LEVEL),
            _ => format!("LEVEL: {}", snap.level),
        }),

        // Sprint frames lines as a countdown to its target; the others count up.
        Metric::Lines => Some(match def.line_target {
            Some(target) => format!("LINES LEFT: {}", target.saturating_sub(snap.lines)),
            None => format!("LINES: {}", snap.lines),
        }),

        // Score is hidden for the time-primary Sprint board (the timer is what
        // matters there); shown for the score-primary variants.
        Metric::Score => match def.score_kind {
            ScoreKind::Score => Some(format!("SCORE: {}", snap.score)),
            ScoreKind::Time => None,
        },

        // "Goal" is the engine's variable-goal remaining-lines-to-next-level
        // figure — only meaningful for Marathon's variable goal system. Sprint's
        // goal is its line target (shown via LINES LEFT) and Ultra's is the
        // clock (shown via TIME), so this row is hidden for them.
        Metric::Goal => match def.end_condition {
            EndCondition::ReachLevel(_) => Some(format!("NEXT LEVEL: {}", snap.goal_remaining)),
            _ => None,
        },

        // Ultra counts the clock down to its limit; the others count elapsed up.
        Metric::Time => Some(match def.time_limit_seconds {
            Some(limit) => format!("TIME LEFT: {}", format_time((limit - elapsed).max(0.0))),
            None => format!("TIME: {}", format_time(elapsed)),
        }),

        // Best result on this variant's board, formatted by its primary key.
        Metric::HighScore => Some(match best {
            Some(entry) => match def.score_kind {
                ScoreKind::Time => format!("BEST: {}", format_time(entry.time_seconds)),
                ScoreKind::Score => format!("BEST: {}", entry.score),
            },
            None => "BEST: --".to_string(),
        }),

        // Optional rates. Both are hidden until there's enough elapsed time to be
        // meaningful (avoids a divide-by-near-zero spike on the first frames).
        Metric::Tpm => {
            rate_per_minute(stats.pieces_locked, elapsed).map(|tpm| format!("TPM: {tpm:.0}"))
        }
        Metric::Lpm => rate_per_minute(snap.lines, elapsed).map(|lpm| format!("LPM: {lpm:.0}")),
    }
}

/// `count` per minute over `elapsed` seconds, or `None` until at least a second
/// has passed (keeps the early-game figure from exploding).
fn rate_per_minute(count: usize, elapsed: f32) -> Option<f32> {
    if elapsed < 1.0 {
        return None;
    }
    Some(count as f32 * 60.0 / elapsed)
}

/// Format seconds as `M:SS` (e.g. `83.4 -> "1:23"`).
fn format_time(seconds: f32) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::high_scores::HighScore;

    fn snapshot_with(
        level: u8,
        lines: usize,
        score: usize,
        goal_remaining: usize,
    ) -> crate::engine::EngineSnapshot {
        let mut snap = Engine::new(EngineConfig::default(), 0).snapshot();
        snap.level = level;
        snap.lines = lines;
        snap.score = score;
        snap.goal_remaining = goal_remaining;
        snap
    }

    fn no_stats() -> RunStats {
        RunStats::default()
    }

    #[test]
    fn format_time_is_minutes_and_padded_seconds() {
        assert_eq!(format_time(0.0), "0:00");
        assert_eq!(format_time(9.0), "0:09");
        assert_eq!(format_time(83.4), "1:23");
        assert_eq!(format_time(125.9), "2:05");
        // Negative inputs (e.g. an overshoot past a time limit) clamp to zero.
        assert_eq!(format_time(-5.0), "0:00");
    }

    #[test]
    fn rate_is_suppressed_until_one_second() {
        assert_eq!(rate_per_minute(10, 0.5), None);
        assert_eq!(rate_per_minute(0, 0.0), None);
        // 30 lines in 60s => 30 LPM.
        assert_eq!(rate_per_minute(30, 60.0), Some(30.0));
    }

    #[test]
    fn marathon_shows_level_climb_score_and_next_level_goal() {
        let def = Variant::Marathon.def();
        let snap = snapshot_with(7, 53, 12_345, 7);

        assert_eq!(
            metric_value(Metric::Level, &def, &snap, 0.0, None, &no_stats()),
            Some(format!("LEVEL: 7/{MARATHON_END_LEVEL}"))
        );
        assert_eq!(
            metric_value(Metric::Lines, &def, &snap, 0.0, None, &no_stats()),
            Some("LINES: 53".to_string())
        );
        assert_eq!(
            metric_value(Metric::Score, &def, &snap, 0.0, None, &no_stats()),
            Some("SCORE: 12345".to_string())
        );
        assert_eq!(
            metric_value(Metric::Goal, &def, &snap, 0.0, None, &no_stats()),
            Some("NEXT LEVEL: 7".to_string())
        );
        // Time counts up for Marathon.
        assert_eq!(
            metric_value(Metric::Time, &def, &snap, 65.0, None, &no_stats()),
            Some("TIME: 1:05".to_string())
        );
    }

    #[test]
    fn sprint_counts_lines_down_hides_score_and_goal() {
        let def = Variant::Sprint.def();
        let snap = snapshot_with(3, 12, 999, 4);

        // 40-line target, 12 cleared => 28 to go.
        assert_eq!(
            metric_value(Metric::Lines, &def, &snap, 0.0, None, &no_stats()),
            Some("LINES LEFT: 28".to_string())
        );
        // Score is hidden for the time-primary board.
        assert_eq!(
            metric_value(Metric::Score, &def, &snap, 0.0, None, &no_stats()),
            None
        );
        // The engine "next level" goal is hidden (Sprint's goal is its lines).
        assert_eq!(
            metric_value(Metric::Goal, &def, &snap, 0.0, None, &no_stats()),
            None
        );
        // Elapsed timer counts up (no limit).
        assert_eq!(
            metric_value(Metric::Time, &def, &snap, 42.0, None, &no_stats()),
            Some("TIME: 0:42".to_string())
        );
    }

    #[test]
    fn sprint_lines_left_saturates_at_zero() {
        let def = Variant::Sprint.def();
        let snap = snapshot_with(3, 45, 0, 0);
        assert_eq!(
            metric_value(Metric::Lines, &def, &snap, 0.0, None, &no_stats()),
            Some("LINES LEFT: 0".to_string())
        );
    }

    #[test]
    fn ultra_counts_time_down_and_shows_score() {
        let def = Variant::Ultra.def();
        let snap = snapshot_with(5, 30, 54_000, 2);

        // 120s limit, 30.5s elapsed => 1:29 remaining.
        assert_eq!(
            metric_value(Metric::Time, &def, &snap, 30.5, None, &no_stats()),
            Some("TIME LEFT: 1:29".to_string())
        );
        assert_eq!(
            metric_value(Metric::Score, &def, &snap, 30.5, None, &no_stats()),
            Some("SCORE: 54000".to_string())
        );
        // Goal row is hidden (Ultra's goal is the clock).
        assert_eq!(
            metric_value(Metric::Goal, &def, &snap, 0.0, None, &no_stats()),
            None
        );
    }

    #[test]
    fn high_score_uses_primary_key_per_variant() {
        let entry = HighScore {
            score: 9000,
            time_seconds: 75.0,
            lines: 40,
            level: 4,
        };
        let snap = snapshot_with(1, 0, 0, 0);

        // Score-primary variants show the score.
        assert_eq!(
            metric_value(
                Metric::HighScore,
                &Variant::Marathon.def(),
                &snap,
                0.0,
                Some(entry),
                &no_stats()
            ),
            Some("BEST: 9000".to_string())
        );
        // Sprint (time-primary) shows the time.
        assert_eq!(
            metric_value(
                Metric::HighScore,
                &Variant::Sprint.def(),
                &snap,
                0.0,
                Some(entry),
                &no_stats()
            ),
            Some("BEST: 1:15".to_string())
        );
        // No entry yet => placeholder.
        assert_eq!(
            metric_value(
                Metric::HighScore,
                &Variant::Marathon.def(),
                &snap,
                0.0,
                None,
                &no_stats()
            ),
            Some("BEST: --".to_string())
        );
    }

    #[test]
    fn tpm_uses_locked_piece_count() {
        let def = Variant::Marathon.def();
        let snap = snapshot_with(1, 20, 0, 0);
        let stats = RunStats { pieces_locked: 60 };
        // 60 pieces in 60s => 60 TPM; 20 lines in 60s => 20 LPM.
        assert_eq!(
            metric_value(Metric::Tpm, &def, &snap, 60.0, None, &stats),
            Some("TPM: 60".to_string())
        );
        assert_eq!(
            metric_value(Metric::Lpm, &def, &snap, 60.0, None, &stats),
            Some("LPM: 20".to_string())
        );
        // Suppressed before 1s elapsed.
        assert_eq!(
            metric_value(Metric::Tpm, &def, &snap, 0.5, None, &stats),
            None
        );
    }
}
