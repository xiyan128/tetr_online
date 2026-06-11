//! Line-clear visual effects (guideline §19.2).
//!
//! Consumes each seat's per-frame [`SeatEvents`](crate::session::SeatEvents)
//! while a session runs and turns them into transient *world-space* flourishes
//! over that seat's playfield:
//!
//! * **Line-clear flash**: a white sheet over the field that pulses and fades on
//!   every line clear.
//! * **Hard-drop trail**: fading vertical streaks down the columns a hard-dropped
//!   piece swept through.
//!
//! The textual callouts ("SINGLE" / "TETRIS" / "T-SPIN" / "COMBO x2" / …) are
//! deliberately *not* here. They live in the session's callout feed
//! ([`crate::session::feel`]), the single home for that text; this module is
//! purely the visual flourish around it.
//!
//! Everything here reads only the engine event stream and never mutates
//! simulation state, so it stays deterministic.

use std::collections::BTreeMap;

use bevy::color::Alpha;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::GameState;
use crate::engine::{EngineEvent, SnapshotCell};
use crate::level::common::to_translation;

/// Lifetime of the white line-clear flash sheet.
const FLASH_TTL_SECONDS: f32 = 0.25;
/// Peak alpha of the line-clear flash. Deliberately gentle — a soft pulse, not a
/// full-board white-out.
const FLASH_PEAK_ALPHA: f32 = 0.18;

/// Lifetime of a hard-drop trail streak.
const TRAIL_TTL_SECONDS: f32 = 0.28;
/// Peak alpha of a hard-drop trail streak.
const TRAIL_PEAK_ALPHA: f32 = 0.4;

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

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

/// Line-clear flash + hard-drop trail visual effects.
pub struct NotificationsPlugin;

impl Plugin for NotificationsPlugin {
    fn build(&self, app: &mut App) {
        // Inspector/scene registration for this feature's transient markers.
        app.register_type::<LineClearFlash>()
            .register_type::<HardDropTrail>()
            // Translate seat events into world effects while a session runs.
            .add_systems(
                Update,
                spawn_event_effects.run_if(in_state(crate::session::SessionPhase::Running)),
            )
            // The fade animators are independent of the engine-step ordering;
            // they keep fading through pause and the result banner.
            .add_systems(
                Update,
                (animate_line_clear_flash, animate_hard_drop_trail)
                    .run_if(in_state(GameState::Session)),
            );
    }
}

// ---------------------------------------------------------------------------
// Engine events -> world effects
// ---------------------------------------------------------------------------

/// Drain each seat's events for the frame and spawn a line-clear flash on any
/// clear plus a hard-drop trail down the columns a hard drop swept through.
///
/// `last_active` caches the active piece's cells from the previous frame so a hard
/// drop — which locks (and replaces) the active piece in the same frame — can
/// still recover the columns/height it swept.
fn spawn_event_effects(
    mut commands: Commands,
    seats: Query<(
        &crate::session::Seat,
        &crate::session::SeatEvents,
        &crate::session::SeatSnapshot,
    )>,
    mut last_active: Local<std::collections::HashMap<usize, Vec<SnapshotCell>>>,
) {
    for (seat, events, snapshot) in &seats {
        let origin = crate::session::render::SessionLayout::board_origin(seat.index);
        let cache = last_active.entry(seat.index).or_default();
        let mut hard_dropped_this_frame = false;

        for event in &events.0 {
            match event {
                EngineEvent::HardDropped { cells_dropped, .. } => {
                    hard_dropped_this_frame = true;
                    spawn_hard_drop_trail(&mut commands, origin, cache.as_slice(), *cells_dropped);
                }
                // Any line-clearing lock pulses the flash.
                EngineEvent::Locked { lines_cleared, .. } if *lines_cleared > 0 => {
                    spawn_line_clear_flash(&mut commands, origin);
                }
                _ => {}
            }
        }

        // Cache the active piece for the *next* frame's potential hard drop.
        // Skip on a hard-drop frame: the snapshot's active piece is already the
        // freshly spawned successor, so caching it would point the next trail
        // at the wrong cells.
        if !hard_dropped_this_frame {
            if let Some(active) = snapshot.0.active.as_ref() {
                cache.clone_from(&active.cells);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Line-clear flash (world space)
// ---------------------------------------------------------------------------

/// Spawn a white sheet covering the visible playfield. Sits just in front of the
/// minos so the clear reads as a soft pulse.
fn spawn_line_clear_flash(commands: &mut Commands, origin: Vec3) {
    use crate::session::render::SessionLayout;
    let width = SessionLayout::BLOCK * SessionLayout::BOARD_W as f32;
    let height = SessionLayout::BLOCK * SessionLayout::BOARD_H as f32;
    commands.spawn((
        LineClearFlash { elapsed: 0.0 },
        Sprite::from_color(
            Color::WHITE.with_alpha(FLASH_PEAK_ALPHA),
            Vec2::new(width, height),
        ),
        // Anchored bottom-left at the SEAT's board origin.
        Transform::from_translation(origin + Vec3::new(0.0, 0.0, 0.5)),
        Anchor::BOTTOM_LEFT,
        DespawnOnExit(GameState::Session),
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
    origin: Vec3,
    start_cells: &[SnapshotCell],
    cells_dropped: usize,
) {
    if start_cells.is_empty() || cells_dropped == 0 {
        return;
    }

    // Per column: lowest pre-drop cell (the leading edge of the streak).
    let mut column_bottom: BTreeMap<isize, (isize, PieceColor)> = BTreeMap::new();
    for cell in start_cells {
        column_bottom
            .entry(cell.x)
            .and_modify(|(min_y, _)| *min_y = (*min_y).min(cell.y))
            .or_insert((cell.y, PieceColor(cell.piece_type)));
    }

    let dropped = cells_dropped as isize;
    let block = crate::session::render::SessionLayout::BLOCK;
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
            Transform::from_translation(origin + Vec3::new(base.x, base.y, -0.05)),
            Anchor::BOTTOM_LEFT,
            DespawnOnExit(GameState::Session),
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
