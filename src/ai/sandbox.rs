//! AI sandbox mode (AI3.6) — watch the bot play, for tuning.
//!
//! The sandbox runs an ordinary gameplay session ([`GameState::Playing`]) whose
//! engine is driven by an [`AiController`] instead of the keyboard. It is the
//! tuning harness for the AI: you see the *real* renderer (the same reconcilers,
//! HUD, SFX, pause overlay, and game-over screen the keyboard game uses) playing
//! against the shipped Tier-1 stack, so difficulty/weights can be judged by eye.
//!
//! # How it is wired (the controller swap, not a parallel renderer)
//!
//! The only thing that differs from a keyboard game is *who polls the engine*.
//! [`LevelPlugin`](crate::level) already steps the engine once per `FixedUpdate`
//! slice; for the keyboard it builds a frame from latched input and polls
//! [`PlayerInput`](crate::level::PlayerInput). This plugin instead:
//!
//! 1. Owns an [`AiSandbox`] flag the menu arms ("Watch AI") or clears ("Play").
//! 2. On entering a gameplay session ([`GameState::Playing`]), when the flag is set,
//!    inserts an [`AiPlayer`] resource holding a fresh [`AiController`].
//! 3. Adds [`step_engine_ai`] to `FixedUpdate`, gated on `Playing` **and** the
//!    flag, which drives the engine through the engine-agnostic
//!    [`drive_engine`](crate::player::drive_engine) seam — the controller emits
//!    its own per-frame `dt`, so the maneuver/neutral cadence is preserved.
//!
//! The keyboard driver ([`step_engine`](crate::level)) is gated on the flag being
//! *unset*, so the two are mutually exclusive and the normal keyboard game runs
//! byte-identically when the sandbox is off. Everything else — board/ghost/HUD
//! reconcilers, audio cues, pause, game over, Esc → menu — is reused unchanged,
//! because the sandbox is just `Playing` with a different controller.
//!
//! # Determinism boundary
//!
//! This is the *only* AI file that imports Bevy. It wraps the pure
//! [`AiController`] in a resource and republishes the engine snapshot/events
//! exactly as the keyboard driver does; the controller still owns its seeded RNG
//! and never touches the engine's generator.

use bevy::prelude::*;

use crate::ai::AiController;
use crate::level::{EngineState, FrameEvents, LatestSnapshot};
use crate::player::drive_engine;
use crate::{GameState, PauseState};

/// Whether the current (or next) gameplay session is the AI sandbox.
///
/// Armed by the "Watch AI" menu entry and cleared by the normal "Play" path, so
/// it is the single switch the level driver reads to decide keyboard vs. AI.
/// Registered for the inspector; a plain `bool` newtype so menu code can flip it
/// without depending on level internals.
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct AiSandbox(pub bool);

impl AiSandbox {
    /// True when the sandbox is armed (the AI should drive the engine).
    pub fn active(self) -> bool {
        self.0
    }
}

/// The bot driving the sandbox session. Inserted on entering gameplay while
/// [`AiSandbox`] is set; dropped when the session ends ([`GameState::Playing`] exits).
///
/// Stored as a **non-send resource** (`NonSend`/`NonSendMut`): [`AiController`] is
/// `Send` but not `Sync` (its [`DecisionRunner`](crate::ai::DecisionRunner) seam is
/// `Send`-only so a future off-thread runner can park a `!Sync` channel/`Task`
/// behind it), and Bevy's ordinary `Resource` requires `Send + Sync`. A non-send
/// resource is the right fit: it pins the bot to the main thread, where the
/// `FixedUpdate` driver already runs. It is also not `Reflect` — like the keyboard
/// [`PlayerInput`](crate::level::PlayerInput) resource it parallels, the
/// engine-agnostic AI core carries no Bevy types.
pub struct AiPlayer(pub AiController);

/// Registers the AI sandbox: the [`AiSandbox`] flag, the per-session controller
/// setup, and the AI engine driver.
pub struct AiSandboxPlugin;

impl Plugin for AiSandboxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AiSandbox>()
            .register_type::<AiSandbox>()
            // Build the controller when a sandbox session starts. Scoped to
            // OnEnter(Playing); pause is a sub-state of Playing, so a pause/resume
            // round-trip keeps the same bot, mirroring the keyboard controller.
            .add_systems(
                OnEnter(GameState::Playing),
                setup_ai_player.run_if(sandbox_active),
            )
            // Drop the bot when the session ends so it never lingers into a later
            // keyboard game and each sandbox run starts from a fresh controller.
            .add_systems(OnExit(GameState::Playing), teardown_ai_player)
            // Drive the engine with the AI once per fixed slice, only while a
            // sandbox game is actually running (frozen while paused — `Running`
            // implies `Playing`). Mutually exclusive with the keyboard
            // `step_engine` (gated on `not sandbox_active`); their conflicting
            // access to the engine resources keeps them ordered.
            .add_systems(
                FixedUpdate,
                step_engine_ai.run_if(in_state(PauseState::Running).and(sandbox_active)),
            );
    }
}

/// Run condition: the AI sandbox is armed. Defaults to `false` when the resource
/// is absent (e.g. a headless `LevelPlugin`-only test), so the keyboard driver
/// stays the default and the normal game is never affected.
pub fn sandbox_active(sandbox: Option<Res<AiSandbox>>) -> bool {
    sandbox.is_some_and(|s| s.active())
}

/// Insert the sandbox's [`AiController`] for this session (beatable default
/// difficulty + the default AI seed). Runs on entering a gameplay session while
/// the sandbox flag is set.
///
/// Exclusive (`&mut World`) because a non-send resource cannot be inserted through
/// [`Commands`] — `insert_non_send_resource` is a `World` method.
fn setup_ai_player(world: &mut World) {
    info!("AI sandbox: bot takes control");
    world.insert_non_send_resource(AiPlayer(AiController::beatable()));
}

/// Remove the sandbox controller when the gameplay session ends (quit to menu,
/// game over → menu). Unconditional: clearing a resource that is already absent
/// (a keyboard session never inserted one) is a no-op, so this stays correct
/// regardless of the flag.
fn teardown_ai_player(world: &mut World) {
    world.remove_non_send_resource::<AiPlayer>();
}

/// Step the engine once per fixed slice by polling the [`AiController`], the
/// AI-sandbox counterpart to the keyboard [`step_engine`](crate::level).
///
/// Uses [`drive_engine`](crate::player::drive_engine): the controller emits the
/// frame to step with (its own `dt` included), so a `dt == 0` maneuver frame
/// positions the piece without advancing gravity and a neutral "thinking" frame
/// advances one sim slice. Unlike the keyboard step we therefore do **not** stamp
/// `time.delta_secs()`. Publishes the snapshot and accumulates events for this
/// frame's reconcilers, exactly like the keyboard driver.
///
/// Fallible: if the controller resource is missing (e.g. the flag was set after
/// the session began) the run condition still let us in, so we no-op rather than
/// panic.
fn step_engine_ai(
    mut engine: ResMut<EngineState>,
    mut snapshot: ResMut<LatestSnapshot>,
    mut frame_events: ResMut<FrameEvents>,
    ai: Option<NonSendMut<AiPlayer>>,
) -> Result {
    let Some(mut ai) = ai else {
        return Ok(());
    };

    let events = drive_engine(&mut engine.0, &mut ai.0);
    snapshot.0 = engine.0.snapshot();
    frame_events.0.extend(events);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::GameAssets;
    use crate::level::LevelPlugin;
    use bevy::time::TimeUpdateStrategy;
    use core::time::Duration;

    /// `sandbox_active` is the seam the keyboard driver negates: it must report the
    /// flag, and — crucially — default to **keyboard** (false) when the resource is
    /// absent, so a headless `LevelPlugin`-only app (no `AiSandboxPlugin`) and the
    /// normal game are never accidentally bot-driven.
    #[test]
    fn sandbox_active_defaults_to_keyboard_when_resource_absent() {
        let mut world = World::new();
        // Absent resource -> keyboard.
        assert!(!world.run_system_cached(sandbox_active).unwrap());

        world.insert_resource(AiSandbox(false));
        assert!(!world.run_system_cached(sandbox_active).unwrap());

        world.insert_resource(AiSandbox(true));
        assert!(world.run_system_cached(sandbox_active).unwrap());
    }

    /// A headless app with the level pipeline **and** the AI sandbox, armed before
    /// the session starts. Mirrors `level::tests::headless_app` (frozen clock so
    /// only explicit `tick_fixed` advances the sim) plus the sandbox plugin.
    fn headless_sandbox_app(arm: bool) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
            .init_state::<GameState>()
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(GameAssets {
                block_texture: default(),
                hard_drop_sound: default(),
                placed_sound: default(),
                line_clear_1: default(),
                line_clear_2: default(),
                line_clear_3: default(),
                line_clear_4: default(),
                locked_sound: default(),
                hold_sound: default(),
                rotation_sound: default(),
                font: default(),
            })
            .add_plugins(LevelPlugin)
            .add_plugins(AiSandboxPlugin);
        // Arm (or clear) the sandbox before entering the session, exactly as the
        // menu does, so `OnEnter(GameState::Playing)` sees the right flag.
        app.insert_resource(AiSandbox(arm));
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);
        app.update(); // queue the transition
        app.update(); // apply Playing + run level + sandbox OnEnter setup
        app
    }

    /// Advance exactly `n` fixed slices through the real schedule (see
    /// `level::tests::tick_fixed`).
    fn tick_fixed(app: &mut App, n: u32) {
        app.insert_resource(TimeUpdateStrategy::FixedTimesteps(n));
        app.update();
    }

    /// With the sandbox armed, entering a session inserts the bot and the AI driver
    /// places pieces with **no keyboard input** — the watch-the-AI contract. Also
    /// confirms the engine actually advances (a piece spawns, then locks).
    #[test]
    fn armed_sandbox_drives_the_engine_with_the_bot() {
        let mut app = headless_sandbox_app(true);

        // The bot resource is created on entering the session.
        assert!(
            app.world().get_non_send_resource::<AiPlayer>().is_some(),
            "an armed sandbox must insert the AiPlayer on session start"
        );

        // One slice spawns the first piece (the controller emits a neutral frame
        // that advances the engine).
        tick_fixed(&mut app, 1);
        assert!(
            app.world().resource::<LatestSnapshot>().0.active.is_some(),
            "the AI driver should advance the engine and spawn a piece"
        );

        // Run enough slices (no keyboard input at all) for the bot to position and
        // hard-drop at least one piece onto the board.
        let mut locked = false;
        for _ in 0..240 {
            tick_fixed(&mut app, 1);
            let cells = &app.world().resource::<LatestSnapshot>().0.board_cells;
            if !cells.is_empty() {
                locked = true;
                break;
            }
        }
        assert!(
            locked,
            "the AI driver must lock a piece with no keyboard input (watch-AI mode)"
        );
    }

    /// With the sandbox **off**, the session inserts no bot and the AI driver does
    /// not run — the keyboard path is untouched. (The keyboard driver itself is
    /// covered by `level`'s tests; here we just assert the sandbox stays inert.)
    #[test]
    fn unarmed_sandbox_inserts_no_bot() {
        let app = headless_sandbox_app(false);
        assert!(
            app.world().get_non_send_resource::<AiPlayer>().is_none(),
            "an unarmed sandbox must not create the AiPlayer"
        );
    }

    /// Leaving the gameplay session (quit / game over → menu) drops the bot so it
    /// can never linger into a later keyboard game.
    #[test]
    fn leaving_the_session_tears_down_the_bot() {
        let mut app = headless_sandbox_app(true);
        assert!(app.world().get_non_send_resource::<AiPlayer>().is_some());

        // Quit to the main menu: exits Playing, firing teardown.
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::MainMenu);
        app.update();
        app.update();

        assert!(
            app.world().get_non_send_resource::<AiPlayer>().is_none(),
            "the bot must be removed when the session ends"
        );
    }
}
