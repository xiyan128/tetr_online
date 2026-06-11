//! Session game feel: seat-aware audio, attack pops, and the garbage slam.
//!
//! Sound design (the ADR's Decision 5): move/rotate/drop SFX play for the
//! **human seat only** — a bot's input stream is noise, and bot-vs-bot would
//! be a drum roll — while line clears play for both seats and a garbage
//! *rise* gets the lock-thunk cue (the thing you must hear is your own board
//! getting heavier). Attack that actually leaves a board shows a brief "+n"
//! pop by the sender's meter. Clears and rises feed the shared screen-shake
//! trauma; the apply mover here is the only camera mover, and it rests the
//! camera at the scene center for however many boards the session shows.

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::transform::TransformSystems;

use crate::GameState;
use crate::assets::GameAssets;
use crate::engine::EngineEvent;
use crate::features::screen_shake::{ScreenShake, trauma_for_clear};
use crate::level::common::{AudioCue, GameplayCamera};

use super::render::SessionLayout;
use super::{HumanSeat, Seat, SeatEvents, SessionPhase};

/// How long a "+n" attack pop lives, and how far it drifts up.
const POP_SECONDS: f32 = 0.8;
const POP_RISE: f32 = 28.0;

pub struct SessionFeelPlugin;

impl Plugin for SessionFeelPlugin {
    fn build(&self, app: &mut App) {
        app
            // Idempotent: `ScreenShakePlugin`/`GamePlugin` own these in the
            // full game; the inits keep a headless versus app self-sufficient.
            .init_resource::<ScreenShake>()
            .init_resource::<crate::vfx::VfxToggles>()
            .add_systems(OnEnter(GameState::Session), reset_session_shake)
            .add_systems(
                Update,
                (emit_seat_audio, spawn_attack_pops, spawn_seat_callouts)
                    .run_if(in_state(SessionPhase::Running)),
            )
            // Same kill-switch as the single-player trauma feed (the dev VFX
            // panel); the apply below keeps running and bleeds to rest.
            .add_systems(
                Update,
                feed_session_trauma
                    .run_if(in_state(SessionPhase::Running).and(crate::vfx::shake_enabled)),
            )
            .add_systems(
                Update,
                (animate_attack_pops, animate_seat_callouts).run_if(in_state(GameState::Session)),
            )
            // Move the camera in PostUpdate before transforms propagate, like
            // the single-player mover — but resting at the two-board center.
            .add_systems(
                PostUpdate,
                apply_session_shake
                    .before(TransformSystems::Propagate)
                    .run_if(in_state(GameState::Session)),
            );
    }
}

fn reset_session_shake(mut shake: ResMut<ScreenShake>) {
    shake.reset();
}

/// Map seat events onto the shared [`AudioCue`] observer. Manoeuvre sounds
/// (rotate / hard drop / hold) are the human seat's only; clears and rises
/// sound for everyone.
fn emit_seat_audio(mut commands: Commands, seats: Query<(&SeatEvents, Option<&HumanSeat>)>) {
    for (events, human) in &seats {
        for event in &events.0 {
            match event {
                // Clears sound for both seats; the no-clear lock thunk is a
                // manoeuvre sound (a bot-vs-bot match would be a metronome).
                EngineEvent::Locked { lines_cleared, .. }
                    if *lines_cleared > 0 || human.is_some() =>
                {
                    commands.trigger(AudioCue::Locked(*lines_cleared));
                }
                // The rise thunk: your board just got heavier.
                EngineEvent::GarbageInserted { .. } => commands.trigger(AudioCue::Placed),
                EngineEvent::Rotated { .. } if human.is_some() => {
                    commands.trigger(AudioCue::Rotation);
                }
                EngineEvent::HardDropped { .. } if human.is_some() => {
                    commands.trigger(AudioCue::HardDrop);
                }
                EngineEvent::Held { .. } if human.is_some() => {
                    commands.trigger(AudioCue::Hold);
                }
                _ => {}
            }
        }
    }
}

/// A clear callout ("TETRIS", "T-SPIN DOUBLE", with a "BACK-TO-BACK" prefix
/// line) in the seat's OUTER side gutter — amber display type that never
/// overlaps the field (Kissaten: the field is sacred).
#[derive(Component)]
struct SeatCallout {
    age: f32,
}

const CALLOUT_TTL: f32 = 1.1;

/// Where a seat's callouts live: just outside the seat's OUTER board edge
/// (away from the shared gutter, the meters, and the opponent), mid-low on
/// the board so a drifting callout never reaches the hold/preview columns.
/// Returns the anchor x and the text anchor to use.
fn callout_anchor(seat: usize) -> (f32, bevy::sprite::Anchor) {
    if seat == 0 {
        (-0.75 * SessionLayout::BLOCK, Anchor::CENTER_RIGHT)
    } else {
        (
            (SessionLayout::BOARD_W as f32 + 0.75) * SessionLayout::BLOCK,
            Anchor::CENTER_LEFT,
        )
    }
}

fn callout_label(action: &crate::engine::EngineScoreAction) -> Option<String> {
    use crate::engine::{EngineScoreAction as A, TSpinKind};
    let core = match action {
        A::Single => "SINGLE".to_string(),
        A::Double => "DOUBLE".to_string(),
        A::Triple => "TRIPLE".to_string(),
        A::Tetris => "TETRIS".to_string(),
        A::TSpin { kind, lines } => {
            let spin = match kind {
                TSpinKind::Mini => "T-SPIN MINI",
                TSpinKind::Full => "T-SPIN",
            };
            match lines {
                0 => spin.to_string(),
                1 => format!("{spin} SINGLE"),
                2 => format!("{spin} DOUBLE"),
                _ => format!("{spin} TRIPLE"),
            }
        }
        _ => return None,
    };
    Some(core)
}

/// Spawn a callout per scoring clear on any seat (both modes — reading the
/// opponent's Tetris matters in versus too). A back-to-back clear gets a
/// smaller "BACK-TO-BACK" prefix line above the core word, both amber.
fn spawn_seat_callouts(
    mut commands: Commands,
    assets: Res<crate::assets::GameAssets>,
    seats: Query<(&Seat, &SeatEvents)>,
) {
    use crate::engine::EngineEvent;
    use crate::ui::widgets::theme;
    for (seat, events) in &seats {
        for event in &events.0 {
            let EngineEvent::ScoreAwarded {
                action,
                back_to_back_bonus,
                ..
            } = event
            else {
                continue;
            };
            let Some(label) = callout_label(action) else {
                continue;
            };
            let origin = SessionLayout::board_origin(seat.index);
            let (anchor_x, anchor) = callout_anchor(seat.index);
            let mid_y = 6.0 * SessionLayout::BLOCK;
            commands.spawn((
                SeatCallout { age: 0.0 },
                Text2d::new(label),
                TextFont {
                    font: assets.font.clone(),
                    font_size: theme::TITLE_FONT_SIZE,
                    ..Default::default()
                },
                TextColor(theme::ACCENT),
                anchor,
                Transform::from_translation(origin + Vec3::new(anchor_x, mid_y, 2.0)),
                DespawnOnExit(GameState::Session),
            ));
            if *back_to_back_bonus {
                commands.spawn((
                    SeatCallout { age: 0.0 },
                    Text2d::new("BACK-TO-BACK"),
                    TextFont {
                        font: assets.font.clone(),
                        font_size: theme::BUTTON_FONT_SIZE,
                        ..Default::default()
                    },
                    TextColor(theme::ACCENT),
                    anchor,
                    Transform::from_translation(origin + Vec3::new(anchor_x, mid_y + 36.0, 2.0)),
                    DespawnOnExit(GameState::Session),
                ));
            }
        }
    }
}

/// Callouts drift up and fade out.
fn animate_seat_callouts(
    mut commands: Commands,
    time: Res<Time>,
    mut callouts: Query<(Entity, &mut SeatCallout, &mut Transform, &mut TextColor)>,
) {
    let dt = time.delta_secs();
    for (entity, mut callout, mut transform, mut color) in &mut callouts {
        callout.age += dt;
        transform.translation.y += 18.0 * dt;
        let life = (callout.age / CALLOUT_TTL).clamp(0.0, 1.0);
        color.0 = color.0.with_alpha(1.0 - life);
        if callout.age >= CALLOUT_TTL {
            commands.entity(entity).despawn();
        }
    }
}

/// A floating "+n" by the sender's meter when net attack leaves a board.
/// `pub(crate)` so the match tests can assert the feedback fires.
#[derive(Component)]
pub(crate) struct AttackPop {
    age: f32,
}

fn spawn_attack_pops(
    mut commands: Commands,
    assets: Res<GameAssets>,
    seats: Query<(&Seat, &SeatEvents)>,
) {
    for (seat, events) in &seats {
        let sent: u32 = events
            .0
            .iter()
            .map(|e| match e {
                EngineEvent::AttackSent { lines } => *lines,
                _ => 0,
            })
            .sum();
        if sent == 0 {
            continue;
        }
        // Next to the seat's meter (the inner board edge), just above the
        // current stack area — world space, scoped to the session.
        let origin = SessionLayout::board_origin(seat.index);
        let x = if seat.index == 0 {
            origin.x + (SessionLayout::BOARD_W as f32 + 1.6) * SessionLayout::BLOCK
        } else {
            origin.x - 1.6 * SessionLayout::BLOCK
        };
        commands.spawn((
            AttackPop { age: 0.0 },
            Text2d::new(format!("+{sent}")),
            TextFont {
                font: assets.font.clone(),
                font_size: crate::ui::widgets::theme::NUMERAL_FONT_SIZE,
                ..default()
            },
            TextColor(crate::ui::widgets::theme::ATTACK),
            Anchor::CENTER,
            Transform::from_translation(Vec3::new(x, 10.0 * SessionLayout::BLOCK, 0.8)),
            DespawnOnExit(GameState::Session),
        ));
    }
}

/// Drift the pops upward and fade them out; retire them at end of life.
fn animate_attack_pops(
    mut commands: Commands,
    time: Res<Time>,
    mut pops: Query<(Entity, &mut AttackPop, &mut Transform, &mut TextColor)>,
) {
    for (entity, mut pop, mut transform, mut color) in &mut pops {
        pop.age += time.delta_secs();
        if pop.age >= POP_SECONDS {
            commands.entity(entity).despawn();
            continue;
        }
        let t = pop.age / POP_SECONDS;
        transform.translation.y += POP_RISE * time.delta_secs() / POP_SECONDS;
        color.0 = color.0.with_alpha(1.0 - t * t);
    }
}

/// Clears thump exactly like single-player; a garbage rise slams in
/// proportion to how many rows just arrived. Both seats feed one trauma —
/// the whole scene shakes, which is right for a shared screen.
fn feed_session_trauma(seats: Query<&SeatEvents>, mut shake: ResMut<ScreenShake>) {
    for events in &seats {
        for event in &events.0 {
            match event {
                EngineEvent::ScoreAwarded {
                    action,
                    back_to_back_bonus,
                    ..
                } => shake.add(trauma_for_clear(*action, *back_to_back_bonus)),
                EngineEvent::GarbageInserted { lines } => {
                    shake.add((0.12 + 0.05 * *lines as f32).min(0.5));
                }
                _ => {}
            }
        }
    }
}

/// The versus camera mover: same noise/decay as single-player (one shared
/// home, `ScreenShake::pose_and_decay`), resting at the two-board center.
fn apply_session_shake(
    time: Res<Time>,
    mut shake: ResMut<ScreenShake>,
    config: Res<super::SessionConfig>,
    mut cameras: Query<&mut Transform, With<GameplayCamera>>,
) {
    let (translation, rotation) = shake.pose_and_decay(
        time.elapsed_secs(),
        time.delta_secs(),
        SessionLayout::scene_center(config.mode.seat_count()),
    );
    for mut transform in &mut cameras {
        transform.translation = translation;
        transform.rotation = rotation;
    }
}
