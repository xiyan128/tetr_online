//! Versus game feel: seat-aware audio, attack pops, and the garbage slam.
//!
//! Sound design (the ADR's Decision 5): move/rotate/drop SFX play for the
//! **human seat only** — a bot's input stream is noise, and bot-vs-bot would
//! be a drum roll — while line clears play for both seats and a garbage
//! *rise* gets the lock-thunk cue (the thing you must hear is your own board
//! getting heavier). Attack that actually leaves a board shows a brief "+n"
//! pop by the sender's meter. Clears and rises feed the shared screen-shake
//! trauma; the versus apply mover rests the camera at the two-board scene
//! center and runs only in `Versus`, so it never fights the single-player
//! mover (gated to `Playing`).

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::transform::TransformSystems;

use crate::assets::GameAssets;
use crate::engine::EngineEvent;
use crate::features::screen_shake::{trauma_for_clear, ScreenShake};
use crate::level::common::{AudioCue, GameplayCamera};
use crate::GameState;

use super::render::SessionLayout;
use super::{HumanSeat, Seat, SeatEvents, SessionPhase};

/// How long a "+n" attack pop lives, and how far it drifts up.
const POP_SECONDS: f32 = 0.8;
const POP_RISE: f32 = 28.0;

pub struct VersusFeelPlugin;

impl Plugin for VersusFeelPlugin {
    fn build(&self, app: &mut App) {
        app
            // Idempotent: `ScreenShakePlugin`/`GamePlugin` own these in the
            // full game; the inits keep a headless versus app self-sufficient.
            .init_resource::<ScreenShake>()
            .init_resource::<crate::vfx::VfxToggles>()
            .add_systems(OnEnter(GameState::Session), reset_versus_shake)
            .add_systems(
                Update,
                (emit_seat_audio, spawn_attack_pops).run_if(in_state(SessionPhase::Running)),
            )
            // Same kill-switch as the single-player trauma feed (the dev VFX
            // panel); the apply below keeps running and bleeds to rest.
            .add_systems(
                Update,
                feed_versus_trauma
                    .run_if(in_state(SessionPhase::Running).and(crate::vfx::shake_enabled)),
            )
            .add_systems(
                Update,
                animate_attack_pops.run_if(in_state(GameState::Session)),
            )
            // Move the camera in PostUpdate before transforms propagate, like
            // the single-player mover — but resting at the two-board center.
            .add_systems(
                PostUpdate,
                apply_versus_shake
                    .before(TransformSystems::Propagate)
                    .run_if(in_state(GameState::Session)),
            );
    }
}

fn reset_versus_shake(mut shake: ResMut<ScreenShake>) {
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
                font_size: 26.0,
                ..default()
            },
            TextColor(Color::srgb_u8(255, 120, 90)),
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
fn feed_versus_trauma(seats: Query<&SeatEvents>, mut shake: ResMut<ScreenShake>) {
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
fn apply_versus_shake(
    time: Res<Time>,
    mut shake: ResMut<ScreenShake>,
    mut cameras: Query<&mut Transform, With<GameplayCamera>>,
) {
    let (translation, rotation) = shake.pose_and_decay(
        time.elapsed_secs(),
        time.delta_secs(),
        SessionLayout::scene_center(),
    );
    for mut transform in &mut cameras {
        transform.translation = translation;
        transform.rotation = rotation;
    }
}
