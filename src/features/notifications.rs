//! Action notifications feature (A1.9, guideline §19.2).
//!
//! Consumes the engine's per-frame [`FrameEvents`] while
//! [`GameState::Playing`](crate::GameState::Playing) and turns them into
//! transient on-screen feedback:
//!
//! * **Toasts** — short labels ("TETRIS", "T-SPIN" / "T-SPIN MINI",
//!   "BACK-TO-BACK", the line-clear count, and a combo counter) that rise in a
//!   top-centre stack and fade out over [`TOAST_TTL_SECONDS`].
//! * **Line-clear flash** — a world-space white sheet over the playfield that
//!   pulses and fades on every line clear.
//! * **Hard-drop trail** — fading vertical streaks down the columns a hard-
//!   dropped piece swept through.
//!
//! A public [`Notification`] message is also exposed so other features can push
//! their own toasts (e.g. "New high score!", "LEVEL UP") without depending on
//! the engine event surface.
//!
//! Styling reuses [`crate::ui::theme`]. The combo counter is derived renderer-
//! side (the engine snapshot carries no combo field) by counting consecutive
//! line-clearing locks; it stays deterministic because it reads only the engine
//! event stream. Nothing here mutates simulation state.
//!
//! Touch only this file.

use bevy::color::Alpha;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::engine::{EngineEvent, EngineScoreAction, SnapshotCell, TSpinKind};
use crate::level::common::{to_translation, LevelConfig, LevelSystems};
use crate::level::engine_bridge::{FrameEvents, LatestSnapshot};
use crate::ui::theme;
use crate::GameState;

/// How long a toast stays fully readable before it begins to fade, plus the
/// total lifetime. The toast holds at full alpha for the first portion of its
/// life then fades to nothing over the remainder.
const TOAST_TTL_SECONDS: f32 = 1.6;
/// Fraction of the lifetime spent fully opaque before the fade begins.
const TOAST_HOLD_FRACTION: f32 = 0.45;
/// Vertical pixels a toast drifts upward over its lifetime.
const TOAST_RISE_PIXELS: f32 = 24.0;

/// Lifetime of the white line-clear flash sheet.
const FLASH_TTL_SECONDS: f32 = 0.32;
/// Peak alpha of the line-clear flash.
const FLASH_PEAK_ALPHA: f32 = 0.5;

/// Lifetime of a hard-drop trail streak.
const TRAIL_TTL_SECONDS: f32 = 0.28;
/// Peak alpha of a hard-drop trail streak.
const TRAIL_PEAK_ALPHA: f32 = 0.4;

// ---------------------------------------------------------------------------
// Public message surface
// ---------------------------------------------------------------------------

/// A request to show a transient toast. Any system can write one:
///
/// ```ignore
/// fn celebrate(mut toasts: MessageWriter<Notification>) {
///     toasts.write(Notification::accent("NEW HIGH SCORE!"));
/// }
/// ```
///
/// The toast renderer (see [`NotificationsPlugin`]) drains these each frame and
/// spawns a fading label. `emphasis` only tweaks colour/size so callers don't
/// have to know about the theme.
#[derive(Message, Clone, Debug)]
pub struct Notification {
    pub text: String,
    pub emphasis: NotificationEmphasis,
}

/// Visual weight of a toast. Maps to a theme colour + font size.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum NotificationEmphasis {
    /// Regular line-clear / combo callouts.
    #[default]
    Normal,
    /// Highlighted callouts (T-Spin, Tetris, Back-to-Back, high score).
    Accent,
}

impl Notification {
    /// A normal-weight toast.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            emphasis: NotificationEmphasis::Normal,
        }
    }

    /// An accent-weight toast (brighter + larger).
    pub fn accent(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            emphasis: NotificationEmphasis::Accent,
        }
    }

    fn color(&self) -> Color {
        match self.emphasis {
            NotificationEmphasis::Normal => theme::TEXT,
            NotificationEmphasis::Accent => theme::ACCENT,
        }
    }

    fn font_size(&self) -> f32 {
        match self.emphasis {
            NotificationEmphasis::Normal => theme::BUTTON_FONT_SIZE,
            NotificationEmphasis::Accent => theme::TITLE_FONT_SIZE,
        }
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// The full-window column that toasts are parented into. Spawned on entering
/// `Playing`, torn down on exit via [`DespawnOnExit`].
#[derive(Component, Reflect)]
#[reflect(Component)]
struct NotificationStack;

/// A live toast label. Tracks its own age so it can fade + rise independently.
#[derive(Component, Reflect)]
#[reflect(Component)]
struct Toast {
    elapsed: f32,
}

/// The world-space white sheet shown on a line clear.
#[derive(Component, Reflect)]
#[reflect(Component)]
struct LineClearFlash {
    elapsed: f32,
}

/// A world-space vertical streak left by a hard drop.
#[derive(Component, Reflect)]
#[reflect(Component)]
struct HardDropTrail {
    elapsed: f32,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Transient on-screen action notifications + line-clear flash + hard-drop
/// trail.
pub struct NotificationsPlugin;

impl Plugin for NotificationsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Notification>()
            // Inspector/scene registration for this feature's transient markers.
            .register_type::<NotificationStack>()
            .register_type::<Toast>()
            .register_type::<LineClearFlash>()
            .register_type::<HardDropTrail>()
            .add_systems(OnEnter(GameState::Playing), spawn_notification_stack)
            // Translate engine events into Notifications + spawn world effects.
            // Runs in the Reconcile set so it sees the same frame's snapshot the
            // other reconcilers do, after the engine driver has stepped.
            .add_systems(
                Update,
                notify_from_engine_events.in_set(LevelSystems::Reconcile),
            )
            // Spawn toasts after the producer so engine-driven callouts appear
            // the same frame; external writers are still picked up next frame.
            .add_systems(
                Update,
                spawn_toasts
                    .after(notify_from_engine_events)
                    .run_if(in_state(GameState::Playing)),
            )
            // The fade animators are independent of the engine-step ordering; run
            // them whenever they have work.
            .add_systems(
                Update,
                (
                    animate_toasts,
                    animate_line_clear_flash,
                    animate_hard_drop_trail,
                )
                    .run_if(in_state(GameState::Playing)),
            );
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Spawn the top-centre toast column. Parented toasts inherit its layout so
/// they stack without each toast computing its own position.
fn spawn_notification_stack(mut commands: Commands) {
    commands.spawn((
        NotificationStack,
        Node {
            position_type: PositionType::Absolute,
            top: percent(12),
            left: px(0),
            width: percent(100),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: px(6),
            ..default()
        },
        DespawnOnExit(GameState::Playing),
    ));
}

// ---------------------------------------------------------------------------
// Engine events -> notifications + world effects
// ---------------------------------------------------------------------------

/// Drain this frame's [`FrameEvents`] and:
///   * push line-clear-count / combo / spin / back-to-back toasts,
///   * spawn a line-clear flash on any clear,
///   * spawn a hard-drop trail down the swept columns.
///
/// `combo` is the renderer-side combo counter (consecutive line-clearing locks).
/// `last_active` caches the active piece's cells from the previous frame so a
/// hard drop — which locks (and replaces) the active piece in the same frame —
/// can still recover the columns/height it swept.
fn notify_from_engine_events(
    mut commands: Commands,
    frame_events: Res<FrameEvents>,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    mut toasts: MessageWriter<Notification>,
    mut combo: Local<u32>,
    mut last_active: Local<Vec<SnapshotCell>>,
) {
    let mut hard_dropped_this_frame = false;

    for event in &frame_events.0 {
        match event {
            EngineEvent::HardDropped { cells_dropped, .. } => {
                hard_dropped_this_frame = true;
                spawn_hard_drop_trail(
                    &mut commands,
                    &config,
                    last_active.as_slice(),
                    *cells_dropped,
                );
            }
            // `Locked` is the authority on how many lines cleared, so it drives
            // the flash + combo bookkeeping. The *callout text* comes from the
            // matching `ScoreAwarded` (it alone knows the T-Spin classification).
            EngineEvent::Locked { lines_cleared, .. } => {
                if *lines_cleared > 0 {
                    *combo += 1;
                    spawn_line_clear_flash(&mut commands, &config);
                    // A combo only reads as such from the second clear onward.
                    if *combo >= 2 {
                        toasts.write(Notification::accent(format!("COMBO x{}", *combo - 1)));
                    }
                } else {
                    *combo = 0;
                }
            }
            EngineEvent::ScoreAwarded {
                action,
                back_to_back_bonus,
                ..
            } => {
                if let Some((label, emphasis)) = clear_label(*action) {
                    toasts.write(Notification {
                        text: label.to_string(),
                        emphasis,
                    });
                }
                if *back_to_back_bonus {
                    toasts.write(Notification::accent("BACK-TO-BACK"));
                }
            }
            _ => {}
        }
    }

    // Cache the active piece for the *next* frame's potential hard drop. Skip on
    // a hard-drop frame: the snapshot's active piece is already the freshly
    // spawned successor, so caching it would point the next trail at the wrong
    // cells. The cache we just consumed stays valid until a real falling piece
    // is observed again.
    if !hard_dropped_this_frame {
        if let Some(active) = snapshot.0.active.as_ref() {
            last_active.clone_from(&active.cells);
        }
    }
}

/// Guideline §19.2 action callout for a scored lock, with its visual emphasis.
///
/// Returns `None` for non-lock awards (soft/hard drop) and plain no-clear locks,
/// which have no banner. T-Spins and Tetrises are accented; ordinary
/// single/double/triple clears use normal weight.
fn clear_label(action: EngineScoreAction) -> Option<(&'static str, NotificationEmphasis)> {
    use NotificationEmphasis::{Accent, Normal};
    let entry = match action {
        EngineScoreAction::Single => ("SINGLE", Normal),
        EngineScoreAction::Double => ("DOUBLE", Normal),
        EngineScoreAction::Triple => ("TRIPLE", Normal),
        EngineScoreAction::Tetris => ("TETRIS", Accent),
        EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines,
        } => match lines {
            0 => ("T-SPIN MINI", Accent),
            _ => ("T-SPIN MINI SINGLE", Accent),
        },
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines,
        } => match lines {
            0 => ("T-SPIN", Accent),
            1 => ("T-SPIN SINGLE", Accent),
            2 => ("T-SPIN DOUBLE", Accent),
            _ => ("T-SPIN TRIPLE", Accent),
        },
        EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. }
        | EngineScoreAction::NoClear => return None,
    };
    Some(entry)
}

// ---------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------

/// Spawn a fading label for each queued [`Notification`], parented to the
/// top-centre stack. If the stack hasn't spawned yet (first frame of `Playing`)
/// the messages are simply dropped — losing a one-frame-early toast is harmless.
fn spawn_toasts(
    mut commands: Commands,
    mut incoming: MessageReader<Notification>,
    assets: Res<crate::assets::GameAssets>,
    // `Option<Single>` rather than `Single`: when the stack hasn't spawned yet the
    // system must still run to drain `incoming` (dropping the early toasts), so it
    // can't be skipped by the scheduler.
    stack: Option<Single<Entity, With<NotificationStack>>>,
) {
    let Some(stack) = stack else {
        incoming.clear();
        return;
    };
    let stack = *stack;

    for notification in incoming.read() {
        let toast = commands
            .spawn((
                Toast { elapsed: 0.0 },
                Text::new(notification.text.clone()),
                TextFont {
                    font: assets.font.clone(),
                    font_size: notification.font_size(),
                    ..default()
                },
                TextColor(notification.color()),
                // Own Node so `animate_toasts` can offset `top` for the rise.
                Node::default(),
            ))
            .id();
        commands.entity(stack).add_child(toast);
    }
}

/// Age every toast: hold at full alpha, then fade, drifting upward the whole
/// time. Despawn once fully transparent.
fn animate_toasts(
    mut commands: Commands,
    time: Res<Time>,
    mut toasts: Query<(Entity, &mut Toast, &mut TextColor, &mut Node)>,
) {
    let dt = time.delta_secs();
    for (entity, mut toast, mut color, mut node) in &mut toasts {
        toast.elapsed += dt;
        let life = (toast.elapsed / TOAST_TTL_SECONDS).clamp(0.0, 1.0);

        // Drift upward (top decreases) as it ages.
        node.top = px(-TOAST_RISE_PIXELS * life);

        let alpha = if life < TOAST_HOLD_FRACTION {
            1.0
        } else {
            let fade = (life - TOAST_HOLD_FRACTION) / (1.0 - TOAST_HOLD_FRACTION);
            (1.0 - fade).clamp(0.0, 1.0)
        };
        color.0 = color.0.with_alpha(alpha);

        if toast.elapsed >= TOAST_TTL_SECONDS {
            commands.entity(entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Line-clear flash (world space)
// ---------------------------------------------------------------------------

/// Spawn a white sheet covering the visible playfield. Sits just in front of the
/// minos so the clear reads as a bright pulse.
fn spawn_line_clear_flash(commands: &mut Commands, config: &LevelConfig) {
    let width = config.block_size * config.board_width as f32;
    let height = config.block_size * config.board_height as f32;
    commands.spawn((
        LineClearFlash { elapsed: 0.0 },
        Sprite::from_color(
            Color::WHITE.with_alpha(FLASH_PEAK_ALPHA),
            Vec2::new(width, height),
        ),
        // Board origin is world (0,0), growing up/right; anchor bottom-left so
        // the sheet lines up with the field.
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.5)),
        Anchor::BOTTOM_LEFT,
        DespawnOnExit(GameState::Playing),
    ));
}

/// Fade the line-clear flash out over its lifetime, then despawn.
fn animate_line_clear_flash(
    mut commands: Commands,
    time: Res<Time>,
    mut flashes: Query<(Entity, &mut LineClearFlash, &mut Sprite)>,
) {
    let dt = time.delta_secs();
    for (entity, mut flash, mut sprite) in &mut flashes {
        flash.elapsed += dt;
        let life = (flash.elapsed / FLASH_TTL_SECONDS).clamp(0.0, 1.0);
        sprite.color = sprite.color.with_alpha(FLASH_PEAK_ALPHA * (1.0 - life));
        if flash.elapsed >= FLASH_TTL_SECONDS {
            commands.entity(entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// Hard-drop trail (world space)
// ---------------------------------------------------------------------------

/// Spawn one fading vertical streak per column the dropped piece occupied,
/// spanning the rows it swept through (`cells_dropped` tall above the landing).
///
/// `start_cells` are the piece's cells *before* the drop (cached last frame).
/// The piece fell straight down by `cells_dropped`, so each column's streak runs
/// from the landing row up to the pre-drop top row.
fn spawn_hard_drop_trail(
    commands: &mut Commands,
    config: &LevelConfig,
    start_cells: &[SnapshotCell],
    cells_dropped: usize,
) {
    if start_cells.is_empty() || cells_dropped == 0 {
        return;
    }

    // Per column: lowest pre-drop cell (the leading edge of the streak).
    use std::collections::BTreeMap;
    let mut column_bottom: BTreeMap<isize, (isize, PieceColor)> = BTreeMap::new();
    for cell in start_cells {
        column_bottom
            .entry(cell.x)
            .and_modify(|(min_y, _)| *min_y = (*min_y).min(cell.y))
            .or_insert((cell.y, PieceColor(cell.piece_type)));
    }

    let dropped = cells_dropped as isize;
    let block = config.block_size;
    let streak_height = (dropped as f32 + 1.0) * block;

    for (x, (bottom_y, color)) in column_bottom {
        // Landing row for this column = pre-drop bottom minus the drop distance.
        let landing_y = bottom_y - dropped;
        let base = to_translation(x, landing_y, block);
        commands.spawn((
            HardDropTrail { elapsed: 0.0 },
            Sprite::from_color(
                crate::level::common::piece_color(color.0).with_alpha(TRAIL_PEAK_ALPHA),
                Vec2::new(block, streak_height),
            ),
            // Behind the minos but in front of the background grid.
            Transform::from_translation(Vec3::new(base.x, base.y, -0.05)),
            Anchor::BOTTOM_LEFT,
            DespawnOnExit(GameState::Playing),
        ));
    }
}

/// Tiny newtype so the per-column map keeps the streak's piece colour.
struct PieceColor(crate::engine::PieceType);

/// Fade each hard-drop streak out over its lifetime, then despawn.
fn animate_hard_drop_trail(
    mut commands: Commands,
    time: Res<Time>,
    mut trails: Query<(Entity, &mut HardDropTrail, &mut Sprite)>,
) {
    let dt = time.delta_secs();
    for (entity, mut trail, mut sprite) in &mut trails {
        trail.elapsed += dt;
        let life = (trail.elapsed / TRAIL_TTL_SECONDS).clamp(0.0, 1.0);
        sprite.color = sprite.color.with_alpha(TRAIL_PEAK_ALPHA * (1.0 - life));
        if trail.elapsed >= TRAIL_TTL_SECONDS {
            commands.entity(entity).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label_text(action: EngineScoreAction) -> Option<&'static str> {
        clear_label(action).map(|(text, _)| text)
    }

    #[test]
    fn plain_line_clears_use_count_labels() {
        assert_eq!(label_text(EngineScoreAction::Single), Some("SINGLE"));
        assert_eq!(label_text(EngineScoreAction::Double), Some("DOUBLE"));
        assert_eq!(label_text(EngineScoreAction::Triple), Some("TRIPLE"));
        assert_eq!(label_text(EngineScoreAction::Tetris), Some("TETRIS"));
    }

    #[test]
    fn t_spin_labels_combine_kind_and_line_count() {
        assert_eq!(
            label_text(EngineScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines: 0,
            }),
            Some("T-SPIN MINI")
        );
        assert_eq!(
            label_text(EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 0,
            }),
            Some("T-SPIN")
        );
        assert_eq!(
            label_text(EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            }),
            Some("T-SPIN DOUBLE")
        );
    }

    #[test]
    fn tetris_and_t_spins_are_accented_plain_clears_are_not() {
        assert_eq!(
            clear_label(EngineScoreAction::Single).map(|(_, e)| e),
            Some(NotificationEmphasis::Normal)
        );
        assert_eq!(
            clear_label(EngineScoreAction::Tetris).map(|(_, e)| e),
            Some(NotificationEmphasis::Accent)
        );
        assert_eq!(
            clear_label(EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 1,
            })
            .map(|(_, e)| e),
            Some(NotificationEmphasis::Accent)
        );
    }

    #[test]
    fn drops_and_no_clear_have_no_banner() {
        assert_eq!(label_text(EngineScoreAction::SoftDrop), None);
        assert_eq!(label_text(EngineScoreAction::HardDrop { cells: 4 }), None);
        assert_eq!(label_text(EngineScoreAction::NoClear), None);
    }
}
