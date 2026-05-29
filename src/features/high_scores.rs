//! High-scores feature (S1.2): record qualifying runs + render the leaderboard.
//!
//! Two halves, both wired here:
//!
//! 1. **Record + persist.** On entering [`GameState::GameOver`] we build a
//!    [`HighScore`] from the final [`LatestSnapshot`] (`score`, `lines`, `level`)
//!    plus [`VariantProgress::elapsed_seconds`], try
//!    [`HighScores::insert`] for the [`ActiveVariant`], and — if it landed on the
//!    board — re-serialize every table and save it through
//!    [`StorageResource`] under [`storage::keys::HIGH_SCORES`]. The board is
//!    loaded back from storage once on startup so leaderboards survive restarts.
//! 2. **Display.** On [`GameState::HighScores`] we populate the screen shell's
//!    [`HighScoresRoot`] with one column per [`Variant`], formatting each row's
//!    primary figure per the variant's [`ScoreKind`] (Sprint shows fastest time
//!    first; Marathon/Ultra show highest score first).
//!
//! The persisted blob is a tiny line-based text format (see [`codec`]) so we keep
//! the shared [`HighScore`] contract untouched and add no serialization
//! dependency. Loading routes every parsed entry back through
//! [`HighScores::insert`], so a corrupt or over-long stored file degrades to a
//! correctly sorted, truncated board rather than a panic.
//!
//! [`GameState::GameOver`]: crate::GameState::GameOver
//! [`GameState::HighScores`]: crate::GameState::HighScores
//! [`HighScore`]: crate::high_scores::HighScore
//! [`HighScores::insert`]: crate::high_scores::HighScores::insert
//! [`HighScores`]: crate::high_scores::HighScores
//! [`LatestSnapshot`]: crate::level::engine_bridge::LatestSnapshot
//! [`VariantProgress::elapsed_seconds`]: crate::variant::VariantProgress
//! [`ActiveVariant`]: crate::variant::ActiveVariant
//! [`StorageResource`]: crate::storage::StorageResource
//! [`HighScoresRoot`]: crate::screens::HighScoresRoot
//! [`Variant`]: crate::variant::Variant
//! [`ScoreKind`]: crate::variant::ScoreKind
//!
//! Touch only this file.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::high_scores::{HighScore, HighScores};
use crate::level::engine_bridge::LatestSnapshot;
use crate::screens::HighScoresRoot;
use crate::storage::{keys, StorageResource};
use crate::ui::widgets::label_text;
use crate::variant::{ActiveVariant, ScoreKind, Variant, VariantProgress};
use crate::GameState;

/// Records qualifying runs into [`HighScores`], persists the table, loads it on
/// startup, and renders the per-variant leaderboard tables.
pub struct HighScoresFeaturePlugin;

impl Plugin for HighScoresFeaturePlugin {
    fn build(&self, app: &mut App) {
        app
            // Load the persisted board once, before any screen reads it. `Startup`
            // runs after `GamePlugin`'s `init_resource::<HighScores>`, so the
            // resource exists; we fill it from storage if a blob is present.
            .add_systems(Startup, load_high_scores)
            // On game over, file the just-finished run and persist on a change.
            .add_systems(OnEnter(GameState::GameOver), record_run)
            // Populate the high-scores screen once its root entity is spawned.
            // Keyed off `Added<HighScoresRoot>` (set by the screen shell on
            // `OnEnter(HighScores)`) so we never depend on `OnEnter` system order.
            .add_systems(
                Update,
                populate_tables.run_if(in_state(GameState::HighScores)),
            );
    }
}

// ---------------------------------------------------------------------------
// Record + persist
// ---------------------------------------------------------------------------

/// Build a [`HighScore`] from the final snapshot + run clock and try to file it
/// for the active variant. Persists the whole board iff the run made the table.
fn record_run(
    snapshot: Res<LatestSnapshot>,
    progress: Res<VariantProgress>,
    active: Res<ActiveVariant>,
    storage: Res<StorageResource>,
    mut scores: ResMut<HighScores>,
) {
    let snap = &snapshot.0;
    let candidate = HighScore {
        score: snap.score,
        time_seconds: progress.elapsed_seconds,
        lines: snap.lines,
        level: snap.level,
    };

    let variant = active.0;
    if let Some(rank) = scores.insert(variant, candidate) {
        info!(
            "high score recorded for {}: rank {} (score {}, {})",
            variant.display_name(),
            rank + 1,
            candidate.score,
            format_time(candidate.time_seconds),
        );
        storage
            .0
            .save(keys::HIGH_SCORES, &codec::serialize(&scores));
    }
}

/// Load the persisted leaderboard on startup, if present. Each entry is routed
/// back through [`HighScores::insert`] so the in-memory board ends up correctly
/// sorted and truncated even if the stored blob was hand-edited or corrupt.
fn load_high_scores(storage: Res<StorageResource>, mut scores: ResMut<HighScores>) {
    let Some(blob) = storage.0.load(keys::HIGH_SCORES) else {
        return;
    };
    let restored = codec::deserialize(&blob);
    *scores = restored;
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// Marker for the table container we attach under [`HighScoresRoot`], so we only
/// build it once per screen visit.
#[derive(Component)]
struct HighScoresTables;

/// Attach one leaderboard column per variant under the screen-shell root.
///
/// Runs each `Update` frame while on the screen but is a no-op unless the root
/// was just added (the shell spawns it on `OnEnter`) and we have not already
/// built the tables for this visit. This sidesteps any `OnEnter` ordering race
/// between this feature and the screen shell.
fn populate_tables(
    mut commands: Commands,
    assets: Res<GameAssets>,
    scores: Res<HighScores>,
    // `Single` skips the system on frames where the root was not just added — the
    // same no-op the early `single()` return used to express.
    root: Single<Entity, Added<HighScoresRoot>>,
    existing: Query<(), With<HighScoresTables>>,
) {
    let root = *root;
    if !existing.is_empty() {
        return;
    }

    let tables = commands
        .spawn((
            HighScoresTables,
            Node {
                flex_direction: FlexDirection::Row,
                column_gap: px(40),
                align_items: AlignItems::FlexStart,
                margin: UiRect::top(px(8)),
                ..default()
            },
        ))
        .id();

    for variant in Variant::ALL {
        let column = spawn_variant_column(&mut commands, &assets, &scores, variant);
        commands.entity(tables).add_child(column);
    }
    commands.entity(root).add_child(tables);
}

/// One vertical column: variant name, a header, then up to ten rows (or an
/// "empty" hint when the board has no entries yet).
fn spawn_variant_column(
    commands: &mut Commands,
    assets: &GameAssets,
    scores: &HighScores,
    variant: Variant,
) -> Entity {
    let font = assets.font.clone();
    let kind = variant.def().score_kind;
    let table = scores.table(variant);

    let column = commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: px(2),
            ..default()
        })
        .id();

    let heading = commands
        .spawn(column_heading(variant.display_name(), font.clone()))
        .id();
    commands.entity(column).add_child(heading);

    let header = commands
        .spawn(label_text(header_row(kind), font.clone()))
        .id();
    commands.entity(column).add_child(header);

    if table.is_empty() {
        let empty = commands
            .spawn(label_text("--- no scores yet ---", font))
            .id();
        commands.entity(column).add_child(empty);
        return column;
    }

    for (index, entry) in table.iter().enumerate() {
        let row = commands
            .spawn(label_text(format_row(index, entry, kind), font.clone()))
            .id();
        commands.entity(column).add_child(row);
    }
    column
}

/// A column heading (variant name), slightly larger / brighter than rows.
fn column_heading(text: &str, font: Handle<Font>) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font,
            font_size: 18.0,
            ..default()
        },
        TextColor(crate::ui::theme::ACCENT),
        Node {
            margin: UiRect::bottom(px(4)),
            ..default()
        },
    )
}

/// Header line naming the primary column for this variant's [`ScoreKind`].
fn header_row(kind: ScoreKind) -> &'static str {
    match kind {
        ScoreKind::Time => "#   TIME       SCORE",
        ScoreKind::Score => "#   SCORE      LINES",
    }
}

/// Format one ranked row. The primary figure (per [`ScoreKind`]) comes first;
/// the secondary figure trails so both are always visible.
fn format_row(index: usize, entry: &HighScore, kind: ScoreKind) -> String {
    let rank = index + 1;
    match kind {
        ScoreKind::Time => format!(
            "{:>2}  {:<9}  {}",
            rank,
            format_time(entry.time_seconds),
            entry.score
        ),
        ScoreKind::Score => format!("{:>2}  {:<9}  {}", rank, entry.score, entry.lines),
    }
}

/// Render seconds as `M:SS.mmm` (e.g. `1:23.456`) so Sprint times read naturally.
fn format_time(seconds: f32) -> String {
    let seconds = seconds.max(0.0);
    let minutes = (seconds / 60.0).floor() as u64;
    let secs = seconds - (minutes as f32) * 60.0;
    format!("{}:{:06.3}", minutes, secs)
}

// ---------------------------------------------------------------------------
// Persistence codec
// ---------------------------------------------------------------------------

/// Line-based, dependency-free encoding for the per-variant boards.
///
/// One entry per line: `<tag> <score> <time_seconds> <lines> <level>`, where
/// `tag` is `M`/`S`/`U` for Marathon/Sprint/Ultra. Lines that don't parse are
/// skipped, and every accepted entry is re-inserted via
/// [`HighScores::insert`], so ordering/truncation are re-derived on load and a
/// malformed file can never corrupt the in-memory board or panic.
mod codec {
    use super::{HighScore, HighScores, Variant};

    fn tag(variant: Variant) -> char {
        match variant {
            Variant::Marathon => 'M',
            Variant::Sprint => 'S',
            Variant::Ultra => 'U',
        }
    }

    fn variant_for_tag(tag: &str) -> Option<Variant> {
        match tag {
            "M" => Some(Variant::Marathon),
            "S" => Some(Variant::Sprint),
            "U" => Some(Variant::Ultra),
            _ => None,
        }
    }

    /// Serialize every variant's table to the line format described above.
    pub fn serialize(scores: &HighScores) -> String {
        let mut out = String::new();
        for variant in Variant::ALL {
            for entry in scores.table(variant) {
                out.push_str(&format!(
                    "{} {} {} {} {}\n",
                    tag(variant),
                    entry.score,
                    entry.time_seconds,
                    entry.lines,
                    entry.level,
                ));
            }
        }
        out
    }

    /// Parse a blob produced by [`serialize`] back into a [`HighScores`],
    /// silently dropping any line that fails to parse.
    pub fn deserialize(blob: &str) -> HighScores {
        let mut scores = HighScores::default();
        for line in blob.lines() {
            if let Some((variant, entry)) = parse_line(line) {
                scores.insert(variant, entry);
            }
        }
        scores
    }

    fn parse_line(line: &str) -> Option<(Variant, HighScore)> {
        let mut fields = line.split_whitespace();
        let variant = variant_for_tag(fields.next()?)?;
        let entry = HighScore {
            score: fields.next()?.parse().ok()?,
            time_seconds: fields.next()?.parse().ok()?,
            lines: fields.next()?.parse().ok()?,
            level: fields.next()?.parse().ok()?,
        };
        // Reject trailing garbage so an extended/forward-incompatible line is
        // skipped rather than silently truncated.
        if fields.next().is_some() {
            return None;
        }
        Some((variant, entry))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn run(score: usize, time: f32, lines: usize, level: u8) -> HighScore {
            HighScore {
                score,
                time_seconds: time,
                lines,
                level,
            }
        }

        #[test]
        fn round_trips_all_variants() {
            let mut scores = HighScores::default();
            scores.insert(Variant::Marathon, run(5000, 90.5, 40, 10));
            scores.insert(Variant::Marathon, run(8000, 120.0, 70, 15));
            scores.insert(Variant::Sprint, run(1200, 42.25, 40, 5));
            scores.insert(Variant::Ultra, run(9999, 120.0, 88, 12));

            let restored = deserialize(&serialize(&scores));

            for variant in Variant::ALL {
                assert_eq!(restored.table(variant), scores.table(variant));
            }
        }

        #[test]
        fn empty_board_round_trips() {
            let scores = HighScores::default();
            let restored = deserialize(&serialize(&scores));
            for variant in Variant::ALL {
                assert!(restored.table(variant).is_empty());
            }
        }

        #[test]
        fn malformed_lines_are_skipped_not_fatal() {
            let blob = "\
M 100 12.5 10 3
garbage line
S notanumber 1 2 3
U 200 7.0 5 4 extrafield
S 1500 30.0 40 6
";
            let restored = deserialize(blob);
            // Only the two well-formed lines survive.
            assert_eq!(restored.table(Variant::Marathon).len(), 1);
            assert_eq!(restored.table(Variant::Marathon)[0].score, 100);
            assert_eq!(restored.table(Variant::Sprint).len(), 1);
            assert_eq!(restored.table(Variant::Sprint)[0].score, 1500);
            assert!(restored.table(Variant::Ultra).is_empty());
        }

        #[test]
        fn deserialize_reorders_and_truncates_unsorted_input() {
            // Sprint ranks by time ascending; feed times out of order and over
            // the cap to prove load re-derives the canonical board.
            let mut blob = String::new();
            for i in 0..15 {
                // times 14.0, 13.0, ... 0.0 (descending input order).
                blob.push_str(&format!("S 0 {}.0 40 1\n", 14 - i));
            }
            let restored = deserialize(&blob);
            let times: Vec<f32> = restored
                .table(Variant::Sprint)
                .iter()
                .map(|e| e.time_seconds)
                .collect();
            assert_eq!(times.len(), 10, "truncated to the per-variant cap");
            assert_eq!(times.first().copied(), Some(0.0), "fastest first");
            assert_eq!(times.last().copied(), Some(9.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_time_renders_minutes_and_millis() {
        assert_eq!(format_time(0.0), "0:00.000");
        assert_eq!(format_time(83.456), "1:23.456");
        assert_eq!(format_time(-5.0), "0:00.000");
    }

    #[test]
    fn time_rows_lead_with_time_score_rows_lead_with_score() {
        let entry = HighScore {
            score: 4200,
            time_seconds: 42.0,
            lines: 40,
            level: 7,
        };
        let time_row = format_row(0, &entry, ScoreKind::Time);
        let score_row = format_row(0, &entry, ScoreKind::Score);
        // Sprint: time precedes score.
        assert!(time_row.contains("0:42.000"));
        assert!(time_row.trim_end().ends_with("4200"));
        // Marathon/Ultra: score precedes lines.
        assert!(score_row.contains("4200"));
        assert!(score_row.trim_end().ends_with("40"));
    }
}
