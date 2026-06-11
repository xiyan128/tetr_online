//! Hit-stop (game feel).
//!
//! A marquee clear — a Tetris or a T-spin — briefly *freezes the whole world*: a
//! few dozen milliseconds of total stillness that the brain reads as weight, the
//! fighting-game "hit-stop" trick. We freeze by pausing Bevy's [`Time<Virtual>`]
//! clock, which in one stroke stalls the `FixedUpdate` engine step (gravity, lock
//! delay, DAS) *and* every virtual-clock flourish (the line-clear flash, the
//! [screen shake](crate::features::screen_shake)) — so the frame holds mid-pose
//! instead of sliding through the impact.
//!
//! The catch the engine docs call out: pausing virtual time also stops
//! `FixedUpdate`, so the *un*pause can't be scheduled on the virtual clock — it
//! would never fire. The countdown therefore runs on [`Time<Real>`], which keeps
//! ticking while the simulation is frozen. As defence in depth the freeze is also
//! force-cleared on every `Session` enter/exit, so a paused clock can never leak
//! across a session boundary and wedge the game.

use bevy::prelude::*;

use crate::engine::{EngineEvent, EngineScoreAction, TSpinKind};
use crate::GameState;

/// Freeze length (seconds of *real* time) for a Tetris.
const FREEZE_TETRIS: f32 = 0.085;
/// Freeze length for a T-spin (the flashiest skill move).
const FREEZE_TSPIN: f32 = 0.10;
/// Extra freeze when the clear extends a back-to-back chain.
const FREEZE_B2B_BONUS: f32 = 0.03;
/// A Tetris/T-spin that *also* extends a long combo shouldn't compound into a
/// visible stall; cap the total freeze here.
const FREEZE_MAX: f32 = 0.16;

/// Tracks an in-progress freeze. `remaining` is real-time seconds left; `0` means
/// the world is running normally.
#[derive(Resource, Default)]
struct HitStop {
    remaining: f32,
}

/// Freezes the simulation for a beat on marquee clears.
pub struct HitStopPlugin;

impl Plugin for HitStopPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HitStop>()
            // Defence in depth: never carry a freeze across a session boundary, so
            // a game can't start — and a menu can't sit — on a paused clock.
            .add_systems(OnEnter(GameState::Session), clear_freeze)
            .add_systems(OnExit(GameState::Session), clear_freeze)
            // Detect marquee clears in the Reconcile set (gated on the
            // session running). The toggle only
            // gates *new* freezes; `tick_hit_stop` always runs so an in-flight
            // freeze still resolves if the effect is switched off mid-freeze.
            .add_systems(
                Update,
                trigger_hit_stop.run_if(
                    in_state(crate::session::SessionPhase::Running)
                        .and(crate::vfx::hit_stop_enabled),
                ),
            )
            // Tick the countdown on the REAL clock so it advances while virtual time
            // is paused. Ordered after the trigger so a freeze begun this frame is
            // timed from its full duration. Gated on `Running` (not the bare
            // session state) so a freeze and a manual pause compose: the freeze
            // holds through the pause menu and resolves on resume, instead of
            // unpausing virtual time behind the overlay. (`clear_freeze` on
            // `Session` exit handles teardown.)
            .add_systems(
                Update,
                tick_hit_stop
                    .after(trigger_hit_stop)
                    .run_if(in_state(crate::session::SessionPhase::Running)),
            );
    }
}

/// Force the world back to running and drop any pending freeze.
fn clear_freeze(mut hit_stop: ResMut<HitStop>, mut virtual_time: ResMut<Time<Virtual>>) {
    hit_stop.remaining = 0.0;
    virtual_time.unpause(); // a no-op if already running
}

/// Start a freeze when this frame produced a Tetris or T-spin. Refreshes (never
/// stacks) the timer so a multi-event frame can't compound into a long stall.
fn trigger_hit_stop(
    config: Res<crate::session::SessionConfig>,
    seats: Query<&crate::session::SeatEvents>,
    mut hit_stop: ResMut<HitStop>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    // Solo only: pausing Time<Virtual> freezes EVERY seat's fixed stepping,
    // which in versus would let one player's Tetris stop the opponent's clock.
    if !matches!(config.mode, crate::session::SessionMode::Solo { .. }) {
        return;
    }
    let mut freeze = 0.0_f32;
    for events in &seats {
        for event in &events.0 {
            if let EngineEvent::ScoreAwarded {
                action,
                back_to_back_bonus,
                ..
            } = event
            {
                freeze = freeze.max(freeze_for_clear(*action, *back_to_back_bonus));
            }
        }
    }
    if freeze > 0.0 {
        hit_stop.remaining = hit_stop.remaining.max(freeze);
        // Takes effect next update (the engine doc's documented one-frame lag),
        // which lands the freeze on the frame the clear's flash is already up.
        virtual_time.pause();
    }
}

/// Freeze duration for a clear, or `0.0` for clears we don't punctuate. Hit-stop
/// is reserved for the marquee moments — singles/doubles/triples get their feel
/// from the shake and the line-clear flash, not a world freeze.
fn freeze_for_clear(action: EngineScoreAction, back_to_back: bool) -> f32 {
    let base = match action {
        EngineScoreAction::Tetris => FREEZE_TETRIS,
        // Any full T-spin (even a 0-line one) is a flashy, deliberate move.
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            ..
        } => FREEZE_TSPIN,
        // A mini T-spin only earns a (shorter) freeze when it actually clears.
        EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines,
        } if lines > 0 => FREEZE_TSPIN * 0.6,
        _ => 0.0,
    };
    if base > 0.0 && back_to_back {
        (base + FREEZE_B2B_BONUS).min(FREEZE_MAX)
    } else {
        base
    }
}

/// Count the freeze down on the real clock and lift it (unpause virtual time) when
/// it elapses. Reading [`Time<Real>`] is what lets this advance while
/// [`Time<Virtual>`] is paused.
fn tick_hit_stop(
    real_time: Res<Time<Real>>,
    mut hit_stop: ResMut<HitStop>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if hit_stop.remaining <= 0.0 {
        return;
    }
    hit_stop.remaining -= real_time.delta_secs();
    if hit_stop.remaining <= 0.0 {
        hit_stop.remaining = 0.0;
        virtual_time.unpause();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;
    use core::time::Duration;

    fn tetris_award() -> EngineEvent {
        EngineEvent::ScoreAwarded {
            action: EngineScoreAction::Tetris,
            score: 800,
            total_score: 800,
            back_to_back_bonus: false,
        }
    }

    #[test]
    fn only_marquee_clears_freeze() {
        assert!(freeze_for_clear(EngineScoreAction::Tetris, false) > 0.0);
        assert!(
            freeze_for_clear(
                EngineScoreAction::TSpin {
                    kind: TSpinKind::Full,
                    lines: 0
                },
                false
            ) > 0.0,
            "even a 0-line full T-spin is a marquee move"
        );
        assert_eq!(freeze_for_clear(EngineScoreAction::Single, false), 0.0);
        assert_eq!(freeze_for_clear(EngineScoreAction::Double, false), 0.0);
        assert_eq!(freeze_for_clear(EngineScoreAction::Triple, false), 0.0);
        assert_eq!(
            freeze_for_clear(
                EngineScoreAction::TSpin {
                    kind: TSpinKind::Mini,
                    lines: 0
                },
                false
            ),
            0.0,
            "a 0-line mini T-spin is too minor to freeze"
        );
    }

    #[test]
    fn back_to_back_lengthens_but_caps_the_freeze() {
        let plain = freeze_for_clear(EngineScoreAction::Tetris, false);
        let b2b = freeze_for_clear(EngineScoreAction::Tetris, true);
        assert!(b2b > plain, "B2B should lengthen the freeze");
        assert!(b2b <= FREEZE_MAX, "freeze must stay under the stall cap");
    }

    /// End-to-end through the real schedule: a Tetris pauses `Time<Virtual>`, and
    /// the freeze lifts once its duration has elapsed on the *real* clock — the
    /// crux of the design (a paused virtual clock can't time its own unpause).
    #[test]
    fn tetris_pauses_virtual_time_then_real_time_resumes_it() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<HitStop>()
            .insert_resource(crate::session::SessionConfig {
                seats: [
                    crate::session::Participant::Human,
                    crate::session::Participant::Bot { model: 0 },
                ],
                mode: crate::session::SessionMode::Solo {
                    variant: crate::variant::Variant::Marathon,
                },
                seed: Some(0),
            })
            // Advance the real clock by a fixed 16 ms per update, independent of
            // wall time, so the countdown is deterministic.
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                16,
            )))
            .add_systems(Update, (trigger_hit_stop, tick_hit_stop).chain());

        // Frame 1: a Tetris award is on a seat's bus → freeze begins.
        let seat = app
            .world_mut()
            .spawn((
                crate::session::Seat { index: 0 },
                crate::session::SeatEvents(vec![tetris_award()]),
            ))
            .id();
        app.update();
        assert!(
            app.world().resource::<Time<Virtual>>().is_paused(),
            "a Tetris must pause the virtual clock"
        );

        // Clear the bus so it can't re-trigger, then let real time march past the
        // freeze length (10 × 16 ms = 160 ms > any freeze).
        app.world_mut()
            .entity_mut(seat)
            .get_mut::<crate::session::SeatEvents>()
            .unwrap()
            .0
            .clear();
        for _ in 0..10 {
            app.update();
        }
        assert!(
            !app.world().resource::<Time<Virtual>>().is_paused(),
            "the freeze must lift after its real-time duration elapses"
        );
    }
}
