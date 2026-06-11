//! Session game feel: seat-aware audio, attack pops, and the garbage slam.
//!
//! Sound design (the ADR's Decision 5): move/rotate/drop SFX play for the
//! **human seat only** — a bot's input stream is noise, and bot-vs-bot would
//! be a drum roll — while line clears play for both seats and a garbage
//! *rise* gets the lock-thunk cue (the thing you must hear is your own board
//! getting heavier). Attack that actually leaves a board shows a brief "+n"
//! pop in the sender's outer gutter, below the callout stack. Clears and
//! rises feed the shared screen-shake trauma; the apply mover here is the
//! only camera mover, and it rests the camera at the scene center for
//! however many boards the session shows.

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

/// Vertical lane of the callout stack's TOP line, in blocks. Lines stack
/// DOWNWARD from here (away from the hold/preview columns above); the
/// "BACK-TO-BACK" prefix sits one short step above. Both extremes stay clear
/// of the columns and the floor — see `callout_lanes_stay_inside_the_gutter`.
const CALLOUT_LANE_BLOCKS: f32 = 5.0;
/// Step between stacked callout lines, px (display type plus leading).
const CALLOUT_LINE_STEP: f32 = 36.0;
/// Vertical lane of the "+n" attack pop, in blocks: under the callout stack,
/// near the gutter floor.
const POP_LANE_BLOCKS: f32 = 2.0;

/// Where a seat's gutter feed lives: just outside the seat's OUTER board edge
/// (away from the shared gutter, the meters, and the opponent). Returns the
/// anchor x (board-relative) and the text anchor to use.
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
/// opponent's Tetris matters in versus too). The label stacks ONE WORD PER
/// LINE down the gutter ("T-SPIN" over "DOUBLE") so the column never grows
/// wider than one display-size word — that worst-case width is what
/// `SessionLayout::scene_min` budgets per side. A back-to-back clear gets a
/// smaller "BACK-TO-BACK" prefix line above the stack, all amber.
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
            let lane_y = CALLOUT_LANE_BLOCKS * SessionLayout::BLOCK;
            let mut spawn_line = |text: &str, size: f32, y: f32| {
                commands.spawn((
                    SeatCallout { age: 0.0 },
                    Text2d::new(text),
                    TextFont {
                        font: assets.font.clone(),
                        font_size: size,
                        ..Default::default()
                    },
                    TextColor(theme::ACCENT),
                    anchor,
                    Transform::from_translation(origin + Vec3::new(anchor_x, y, 2.0)),
                    DespawnOnExit(GameState::Session),
                ));
            };
            for (line, word) in label.split_whitespace().enumerate() {
                spawn_line(
                    word,
                    theme::TITLE_FONT_SIZE,
                    lane_y - line as f32 * CALLOUT_LINE_STEP,
                );
            }
            if *back_to_back_bonus {
                spawn_line(
                    "BACK-TO-BACK",
                    theme::BUTTON_FONT_SIZE,
                    lane_y + 0.85 * CALLOUT_LINE_STEP,
                );
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

/// A floating "+n" in the sender's outer gutter when net attack leaves a
/// board. Amber, not attack red — red is reserved for INCOMING pressure (the
/// opponent reads this seat's outgoing attack on their own meter); sending is
/// a positive moment and joins the gutter feed below the callout stack.
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
        let origin = SessionLayout::board_origin(seat.index);
        let (anchor_x, anchor) = callout_anchor(seat.index);
        commands.spawn((
            AttackPop { age: 0.0 },
            Text2d::new(format!("+{sent}")),
            TextFont {
                font: assets.font.clone(),
                font_size: crate::ui::widgets::theme::NUMERAL_FONT_SIZE,
                ..default()
            },
            TextColor(crate::ui::widgets::theme::ACCENT),
            anchor,
            Transform::from_translation(
                origin + Vec3::new(anchor_x, POP_LANE_BLOCKS * SessionLayout::BLOCK, 0.8),
            ),
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

#[cfg(test)]
mod gutter_tests {
    use super::*;

    /// Dogica is strictly monospaced at 1.0 em per glyph, so a word's world
    /// width is `glyphs * font_size`.
    const GLYPH_EM: f32 = 1.0;

    /// Every word the callout feed can emit, across the whole engine
    /// vocabulary (plus the B2B prefix, which renders at the label size).
    fn callout_words() -> Vec<String> {
        use crate::engine::{EngineScoreAction as A, TSpinKind};
        let mut actions = vec![A::Single, A::Double, A::Triple, A::Tetris];
        for kind in [TSpinKind::Mini, TSpinKind::Full] {
            for lines in 0..=3 {
                actions.push(A::TSpin { kind, lines });
            }
        }
        actions
            .iter()
            .filter_map(callout_label)
            .flat_map(|label| {
                label
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// How much horizontal room `scene_min` guarantees outboard of a board's
    /// outer edge, past the callout anchor.
    fn gutter_room(seat_count: usize) -> f32 {
        let (min_width, _) = SessionLayout::scene_min(seat_count);
        let board_span = SessionLayout::board_origin(seat_count.saturating_sub(1)).x
            + SessionLayout::BOARD_W as f32 * SessionLayout::BLOCK;
        (min_width - board_span) / 2.0 - 0.75 * SessionLayout::BLOCK
    }

    #[test]
    fn every_callout_word_fits_the_guaranteed_camera_framing() {
        let widest = callout_words()
            .iter()
            .map(|w| w.len() as f32 * GLYPH_EM * crate::ui::widgets::theme::TITLE_FONT_SIZE)
            .fold(0.0f32, f32::max);
        let prefix =
            "BACK-TO-BACK".len() as f32 * GLYPH_EM * crate::ui::widgets::theme::BUTTON_FONT_SIZE;
        for seats in [1, 2] {
            let room = gutter_room(seats);
            assert!(
                widest <= room && prefix <= room,
                "a {seats}-seat scene guarantees {room}px of gutter; the feed needs \
                 {widest}px (words) / {prefix}px (prefix) — widen scene_min"
            );
        }
    }

    #[test]
    fn the_gutter_feed_lanes_clear_the_preview_column_and_the_floor() {
        let block = SessionLayout::BLOCK;
        // Worst-case preview column: MAX_NEXT_COUNT two-row avatars + gaps.
        let avatar = 2.0 * block * SessionLayout::PREVIEW_SCALE;
        let gap = 0.5 * block * SessionLayout::PREVIEW_SCALE;
        let count = crate::settings::MAX_NEXT_COUNT as f32;
        let preview_bottom =
            SessionLayout::BOARD_H as f32 * block - (count * avatar + (count - 1.0) * gap);

        let drift = 18.0 * CALLOUT_TTL;
        let prefix_top = CALLOUT_LANE_BLOCKS * block
            + 0.85 * CALLOUT_LINE_STEP
            + crate::ui::widgets::theme::BUTTON_FONT_SIZE / 2.0
            + drift;
        assert!(
            prefix_top < preview_bottom,
            "the callout stack's top ({prefix_top}) must clear the preview column \
             ({preview_bottom})"
        );

        let pop_top =
            POP_LANE_BLOCKS * block + POP_RISE + crate::ui::widgets::theme::NUMERAL_FONT_SIZE / 2.0;
        assert!(pop_top < preview_bottom);

        // The deepest stack (three words) stays above the board floor.
        let deepest = CALLOUT_LANE_BLOCKS * block
            - 2.0 * CALLOUT_LINE_STEP
            - crate::ui::widgets::theme::TITLE_FONT_SIZE / 2.0;
        assert!(
            deepest > 0.0,
            "the third callout line sinks below the floor"
        );
    }
}
