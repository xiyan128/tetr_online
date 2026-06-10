//! Trauma-based screen shake (game feel).
//!
//! A single [`ScreenShake`] trauma value in `[0, 1]` is bumped by impactful engine
//! events — line clears scaled by severity, and hard drops — and bleeds off
//! linearly. Each frame the camera is displaced by smooth value-noise scaled by
//! `trauma²`, the canonical Squirrel-Eiserloh design (GDC 2016): the squared
//! response ramps in hard and tails off gently, so the shake reads as *impact*
//! rather than a constant buzz.
//!
//! Only the [`GameplayCamera`] is moved, and the system always rewrites it from
//! its rest [`camera_center`] — so the effect is fully self-contained (nothing
//! else touches the camera transform) and a calm board leaves it exactly centered
//! and level. Trauma and the noise phase ride the *virtual* clock, so a hit-stop
//! freeze ([`crate::features::hit_stop`]) holds the camera mid-shake for a punchy
//! frozen frame instead of sliding through it.
//!
//! The noise is hand-rolled and deterministic (no RNG resource, no dependency), so
//! it stays reproducible and headless-test-friendly, and works identically on
//! every render backend — it never leaves the CPU.

use bevy::prelude::*;
use bevy::transform::TransformSystems;

use crate::engine::{EngineEvent, EngineScoreAction, TSpinKind};
use crate::level::common::{camera_center, GameplayCamera, LevelConfig, LevelSystems};
use crate::level::FrameEvents;
use crate::GameState;

/// How fast trauma bleeds off, in trauma-units per second. Tuned so a full
/// game-over jolt settles in well under a second and a single-line tap is gone in
/// a blink.
const TRAUMA_DECAY_PER_SEC: f32 = 1.4;
/// Trauma is raised to this power before driving the offset: low trauma barely
/// stirs the camera, high trauma slams it. The non-linearity is what makes the
/// shake feel weighty.
const TRAUMA_EXPONENT: f32 = 2.0;
/// Peak camera translation (world units ≈ px at this zoom) at full shake.
const MAX_OFFSET: f32 = 14.0;
/// Peak camera roll (radians) at full shake. Deliberately tiny — a hint of roll
/// reads as impact; more reads as nausea.
const MAX_ANGLE: f32 = 0.018;
/// How fast the noise field scrolls. Higher is buzzier, lower is wobblier.
const NOISE_FREQUENCY: f32 = 24.0;

/// Current screen-shake trauma in `[0, 1]`. Bumped by [`accumulate_trauma`] and
/// bled off by [`apply_shake`].
#[derive(Resource, Default, Reflect)]
#[reflect(Resource)]
pub struct ScreenShake {
    trauma: f32,
}

impl ScreenShake {
    /// Add `amount` of trauma, saturating at the `1.0` ceiling. `pub(crate)`:
    /// the versus mode feeds the same resource from its own per-seat events
    /// (its apply runs only in `Versus`, this module's only in `Playing`, so
    /// the two movers never fight over a camera).
    pub(crate) fn add(&mut self, amount: f32) {
        self.trauma = (self.trauma + amount).clamp(0.0, 1.0);
    }

    /// Zero the trauma (a fresh session starts calm).
    pub(crate) fn reset(&mut self) {
        self.trauma = 0.0;
    }

    /// Camera pose for the current trauma around `rest`, and bleed trauma off
    /// by `dt`. The one home of the noise/decay math, shared by the
    /// single-player and versus apply systems (each gated to its own state).
    pub(crate) fn pose_and_decay(
        &mut self,
        elapsed_secs: f32,
        dt: f32,
        rest: Vec3,
    ) -> (Vec3, Quat) {
        let amount = self.trauma.powf(TRAUMA_EXPONENT);
        let pose = if amount <= f32::EPSILON {
            (rest, Quat::IDENTITY)
        } else {
            let phase = elapsed_secs * NOISE_FREQUENCY;
            let dx = value_noise(phase) * amount * MAX_OFFSET;
            let dy = value_noise(phase + 137.0) * amount * MAX_OFFSET;
            let roll = value_noise(phase + 731.0) * amount * MAX_ANGLE;
            (rest + Vec3::new(dx, dy, 0.0), Quat::from_rotation_z(roll))
        };
        self.trauma = (self.trauma - TRAUMA_DECAY_PER_SEC * dt).max(0.0);
        pose
    }
}

/// Drives the [`ScreenShake`] resource and the gameplay camera offset.
pub struct ScreenShakePlugin;

impl Plugin for ScreenShakePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenShake>()
            .register_type::<ScreenShake>()
            // A fresh game always starts calm, even if the last one ended mid-shake
            // (the resource persists across sessions).
            .add_systems(OnEnter(GameState::Playing), reset_shake)
            // Read this frame's engine events in the Reconcile set — gated on
            // `PauseState::Running`, after the engine has stepped — exactly like
            // the other event-driven effects. When the toggle is off we stop
            // feeding trauma; `apply_shake` keeps running and bleeds the camera
            // back to rest.
            .add_systems(
                Update,
                accumulate_trauma
                    .in_set(LevelSystems::Reconcile)
                    .run_if(crate::vfx::shake_enabled),
            )
            // Move the camera in PostUpdate, before transform propagation feeds the
            // offset to the renderer. Runs whenever a game is live (not gated on
            // pause) so any residual jolt always settles back to center.
            .add_systems(
                PostUpdate,
                apply_shake
                    .before(TransformSystems::Propagate)
                    .run_if(in_state(GameState::Playing)),
            );
    }
}

/// Zero the trauma so each new game starts from a still camera.
fn reset_shake(mut shake: ResMut<ScreenShake>) {
    shake.reset();
}

/// Bump trauma from this frame's engine events. Severity scales with how
/// impactful the moment is: a Tetris or T-spin jolts hard, a single barely stirs,
/// a hard drop that actually travels adds a small kick.
///
/// Game over is deliberately *not* shaken here: the gameplay camera is despawned on
/// the same `Playing -> GameOver` transition, so a jolt would render for at most one
/// (invisible) frame — game-over juice belongs to the game-over screen.
fn accumulate_trauma(frame_events: Res<FrameEvents>, mut shake: ResMut<ScreenShake>) {
    for event in &frame_events.0 {
        match event {
            EngineEvent::ScoreAwarded {
                action,
                back_to_back_bonus,
                ..
            } => shake.add(trauma_for_clear(*action, *back_to_back_bonus)),
            // A small kick that grows with the drop distance but stays modest — hard
            // drops are constant, so a big shake here would fatigue. A zero-distance
            // drop (tapping hard-drop on an already-grounded piece) gets no kick.
            EngineEvent::HardDropped { cells_dropped, .. } if *cells_dropped > 0 => {
                shake.add((0.06 + 0.008 * *cells_dropped as f32).min(0.18));
            }
            _ => {}
        }
    }
}

/// Trauma contributed by a scoring clear. Drop / no-clear actions contribute
/// nothing here (their feel comes from the hard-drop kick instead).
/// `pub(crate)`: the versus feel pass scales clears identically.
pub(crate) fn trauma_for_clear(action: EngineScoreAction, back_to_back: bool) -> f32 {
    let base = match action {
        EngineScoreAction::Single => 0.16,
        EngineScoreAction::Double => 0.26,
        EngineScoreAction::Triple => 0.40,
        EngineScoreAction::Tetris => 0.62,
        EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines,
        } => {
            if lines == 0 {
                0.22
            } else {
                0.38
            }
        }
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines,
        } => 0.48 + 0.12 * lines as f32,
        EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. }
        | EngineScoreAction::NoClear => 0.0,
    };
    // Back-to-back is a flourish: give qualifying clears a touch more punch.
    if back_to_back && base > 0.0 {
        (base + 0.10).min(1.0)
    } else {
        base
    }
}

/// Offset the gameplay camera by `trauma²`-scaled smooth noise, then bleed off
/// trauma. Always rewrites the camera from its rest center, so zero trauma means a
/// perfectly centered, level camera.
fn apply_shake(
    time: Res<Time>,
    config: Res<LevelConfig>,
    mut shake: ResMut<ScreenShake>,
    mut cameras: Query<&mut Transform, With<GameplayCamera>>,
) {
    // Compute the pose once, then write it to the gameplay camera(s). Iterating
    // (rather than `single_mut`) keeps trauma bleeding off even on the frames the
    // camera is briefly absent, and never leaves a stray camera stuck off-center.
    // (Decay happens inside `pose_and_decay`, after the pose, so a fresh jolt
    // shows at full strength this frame.)
    let (translation, rotation) = shake.pose_and_decay(
        time.elapsed_secs(),
        time.delta_secs(),
        camera_center(&config),
    );
    for mut transform in &mut cameras {
        transform.translation = translation;
        transform.rotation = rotation;
    }
}

// ---------------------------------------------------------------------------
// Smooth value noise (dependency-free, deterministic)
// ---------------------------------------------------------------------------

/// Smooth 1-D value noise in `[-1, 1]`: hash the integer lattice to pseudo-random
/// amplitudes and smoothstep-interpolate between neighbours. Deterministic, so the
/// shake is reproducible and needs no RNG resource.
fn value_noise(x: f32) -> f32 {
    let i = x.floor();
    let f = x - i;
    let lattice = i as i32;
    let a = lattice_amplitude(lattice);
    let b = lattice_amplitude(lattice + 1);
    let u = f * f * (3.0 - 2.0 * f); // smoothstep ease
    a + (b - a) * u
}

/// Hash a lattice index to a stable pseudo-random amplitude in `[-1, 1]` via an
/// integer avalanche mix.
fn lattice_amplitude(n: i32) -> f32 {
    let mut h = n as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trauma_saturates_at_one() {
        let mut shake = ScreenShake::default();
        shake.add(0.7);
        shake.add(0.7);
        assert_eq!(shake.trauma, 1.0, "trauma must clamp at the 1.0 ceiling");
    }

    #[test]
    fn bigger_clears_shake_harder() {
        let single = trauma_for_clear(EngineScoreAction::Single, false);
        let tetris = trauma_for_clear(EngineScoreAction::Tetris, false);
        let tspin = trauma_for_clear(
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            },
            false,
        );
        assert!(single > 0.0);
        assert!(tetris > single, "a Tetris must out-shake a single");
        assert!(tspin > single, "a T-spin double must out-shake a single");
    }

    #[test]
    fn drops_and_no_clear_contribute_no_clear_trauma() {
        assert_eq!(trauma_for_clear(EngineScoreAction::SoftDrop, false), 0.0);
        assert_eq!(
            trauma_for_clear(EngineScoreAction::HardDrop { cells: 9 }, false),
            0.0
        );
        assert_eq!(trauma_for_clear(EngineScoreAction::NoClear, false), 0.0);
    }

    #[test]
    fn back_to_back_adds_punch_only_to_scoring_clears() {
        assert!(
            trauma_for_clear(EngineScoreAction::Tetris, true)
                > trauma_for_clear(EngineScoreAction::Tetris, false),
            "B2B should add punch to a qualifying clear"
        );
        assert_eq!(
            trauma_for_clear(EngineScoreAction::NoClear, true),
            0.0,
            "B2B must not conjure trauma out of a non-scoring action"
        );
    }

    #[test]
    fn value_noise_stays_in_unit_range_and_is_continuous() {
        // Sample densely; the output must never leave [-1, 1].
        let mut x = -50.0;
        while x < 50.0 {
            let v = value_noise(x);
            assert!((-1.0..=1.0).contains(&v), "noise out of range at {x}: {v}");
            x += 0.013;
        }
        // At integer inputs the interpolation collapses onto the lattice value, so
        // crossing a lattice boundary is continuous (no jumps).
        for k in -5..5 {
            let at = value_noise(k as f32);
            let just_before = value_noise(k as f32 - 1e-4);
            assert!(
                (at - just_before).abs() < 1e-2,
                "noise should be continuous across lattice index {k}"
            );
        }
    }
}
