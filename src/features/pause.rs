//! Pause feature (A1.2).
//!
//! While [`GameState::Playing`](crate::GameState::Playing), pressing the bound
//! Pause action toggles to [`GameState::Paused`](crate::GameState::Paused) and
//! back. The level plugin gates its driver + reconcilers on `Playing`, so simply
//! being in `Paused` halts the sim with no extra work — the authoritative engine
//! lives in resources (untouched by the state change), so resuming continues
//! exactly where it left off (no desync, no reseed).
//!
//! On `Paused` this feature:
//! * hides the Matrix / Next / Hold *contents* (the playfield blocks and the
//!   preview/hold minos) so the board is concealed behind the overlay, then
//!   restores them on resume, and
//! * draws a keyboard-navigable "PAUSE" overlay offering **Resume** / **Quit**.
//!
//! ## How pause is modeled
//! Pause is the [`PauseState`](crate::PauseState) **sub-state of
//! [`GameState::Playing`]** (Running/Paused), not a sibling `GameState`. Toggling
//! it never exits `Playing`, so the session — engine, board, camera, and HUD, all
//! scoped to `OnEnter(GameState::Playing)` / `DespawnOnExit(GameState::Playing)` —
//! survives a pause/resume round-trip with no rebuild. The level's per-frame
//! driver/reconciler/UI systems gate on `PauseState::Running`, so the simulation
//! freezes while paused without anything being despawned. (An earlier design used
//! a `Playing | Paused` *computed* state for the session scope, but a
//! `ComputedStates` re-runs OnEnter/OnExit on every source change — restarting the
//! game on each pause; the sub-state avoids that.)

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::level::common::{BackgroundBlock, FallingBlock, GhostBlock, PreviewBlock, StaticBlock};
use crate::settings::{GameAction, GameSettings};
use crate::ui::focus::{focus_navigation, read_nav_action, FocusList, Focusable, NavAction};
use crate::ui::widgets::{label_text, menu_button, screen_root, title_text};
use crate::{GameState, PauseState};

/// Pause overlay + `Playing <-> Paused` toggle.
pub struct PausePlugin;

impl Plugin for PausePlugin {
    fn build(&self, app: &mut App) {
        app
            // Inspector/scene registration for this feature's markers.
            .register_type::<PauseRoot>()
            .register_type::<PauseAction>()
            // Pause from gameplay on the Pause keybind (only while running).
            .add_systems(
                Update,
                toggle_to_paused.run_if(in_state(PauseState::Running)),
            )
            // Build the overlay on pause; DespawnOnExit(PauseState::Paused) tears it
            // down on resume — and also when the parent `Playing` exits (Quit).
            .add_systems(OnEnter(PauseState::Paused), setup_pause_overlay)
            // Conceal the playfield iff paused, re-evaluated EVERY frame in a
            // session. Idempotent (not a one-shot OnEnter-hide / OnExit-show pair),
            // so rapid toggling can never leave the board stuck hidden or shown —
            // visibility always matches the live sub-state.
            .add_systems(
                Update,
                sync_gameplay_visibility.run_if(in_state(GameState::Playing)),
            )
            // Drive the overlay's focus + selection while paused.
            .add_systems(
                Update,
                (focus_navigation::<PauseRoot>, activate)
                    .chain()
                    .run_if(in_state(PauseState::Paused)),
            );
    }
}

/// Menu rows on the pause overlay, in display (focus) order.
#[derive(Component, Clone, Copy, Reflect)]
#[reflect(Component)]
enum PauseAction {
    Resume,
    Quit,
}

const ITEMS: [(PauseAction, &str); 2] = [
    (PauseAction::Resume, "Resume"),
    (PauseAction::Quit, "Quit to Menu"),
];

/// Marker for the pause overlay's screen-root (carries the [`FocusList`]).
#[derive(Component, Reflect)]
#[reflect(Component)]
struct PauseRoot;

/// True on the rising edge of the bound Pause action (primary or secondary key).
fn pause_just_pressed(keys: &ButtonInput<KeyCode>, settings: &GameSettings) -> bool {
    let (primary, secondary) = settings.keybinds.get(GameAction::Pause);
    keys.just_pressed(primary) || secondary.is_some_and(|key| keys.just_pressed(key))
}

/// While running, the Pause keybind enters the [`PauseState::Paused`] sub-state.
fn toggle_to_paused(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<GameSettings>,
    mut pause: ResMut<NextState<PauseState>>,
) {
    if pause_just_pressed(&keys, &settings) {
        pause.set(PauseState::Paused);
    }
}

/// Spawn the centered "PAUSE" overlay with a Resume / Quit menu, reusing the
/// shared widgets + focus-navigation contract. Despawns on leaving `Paused`.
fn setup_pause_overlay(mut commands: Commands, assets: Res<GameAssets>) {
    let root = commands
        .spawn((
            PauseRoot,
            FocusList::new(ITEMS.len()),
            screen_root(),
            // Dim the (hidden) board behind the overlay for contrast.
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
            DespawnOnExit(PauseState::Paused),
            children![
                title_text("PAUSE", assets.font.clone()),
                label_text("Enter to select  -  Esc to resume", assets.font.clone()),
            ],
        ))
        .id();

    for (index, (action, label)) in ITEMS.into_iter().enumerate() {
        let button = commands
            .spawn((menu_button(index, label, assets.font.clone()), action))
            .id();
        commands.entity(root).add_child(button);
    }
}

/// Resume (Enter/Space on Resume, or Esc / the Pause key) and Quit (Enter/Space
/// on Quit). Esc always resumes — it doubles as the natural "unpause" key.
fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<GameSettings>,
    lists: Query<&FocusList, With<PauseRoot>>,
    actions: Query<(&Focusable, &PauseAction)>,
    mut pause: ResMut<NextState<PauseState>>,
    mut game: ResMut<NextState<GameState>>,
) {
    // The Pause keybind (when it isn't Escape) also resumes, mirroring the
    // Running-side toggle so the same key pauses and unpauses.
    let (primary, _) = settings.keybinds.get(GameAction::Pause);
    if primary != KeyCode::Escape && pause_just_pressed(&keys, &settings) {
        pause.set(PauseState::Running);
        return;
    }

    // `iter().next()` rather than `single()`: during fast toggling there can be a
    // transient frame with zero or more-than-one overlay root, and we must not
    // get stuck unable to resume in that frame.
    let Some(list) = lists.iter().next() else {
        return;
    };
    match read_nav_action(&keys, list) {
        // Resume = re-enter the Running sub-state (stays in Playing). Quit = leave
        // Playing entirely, which despawns the session and the overlay.
        Some(NavAction::Back) => pause.set(PauseState::Running),
        Some(NavAction::Select(index)) => {
            for (focusable, action) in &actions {
                if focusable.index != index {
                    continue;
                }
                match action {
                    PauseAction::Resume => pause.set(PauseState::Running),
                    PauseAction::Quit => game.set(GameState::MainMenu),
                }
            }
        }
        None => {}
    }
}

/// All playfield-content marker components — the Matrix grid + locked/active/
/// ghost minos and the Next/Hold preview minos. Used to gate visibility while
/// paused. (Excludes score/line text, which sit beside the board.)
type GameplayContent = Or<(
    With<BackgroundBlock>,
    With<StaticBlock>,
    With<FallingBlock>,
    With<GhostBlock>,
    With<PreviewBlock>,
)>;

/// Conceal the Matrix / Next / Hold contents iff the game is paused.
///
/// Runs EVERY frame while a gameplay session is alive (Playing or Paused) and
/// derives visibility from the current state. This is deliberately idempotent: a
/// one-shot OnEnter-hide / OnExit-show pair desyncs under rapid Esc toggling — a
/// missed or coalesced transition leaves the static grid stuck `Hidden`, i.e. an
/// "empty board" that never recovers. Re-deriving visibility from the live state
/// each frame is self-correcting. Entities survive the transition (session-
/// scoped), so we hide rather than despawn; the engine snapshot is preserved for
/// a clean resume, and the reconcilers repaint pieces on the next Playing frame.
fn sync_gameplay_visibility(
    pause: Res<State<PauseState>>,
    mut content: Query<&mut Visibility, GameplayContent>,
) {
    let desired = if *pause.get() == PauseState::Paused {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut visibility in &mut content {
        if *visibility != desired {
            *visibility = desired;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::GameAssets;
    use crate::engine::InputFrame;
    use crate::level::{EngineState, LevelPlugin};

    fn test_assets() -> GameAssets {
        GameAssets {
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
        }
    }

    fn count_cameras(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<(), With<Camera2d>>()
            .iter(app.world())
            .count()
    }

    fn set_state(app: &mut App, state: GameState) {
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(state);
        app.update(); // queue the transition
        app.update(); // apply it + run OnEnter/OnExit
    }

    fn set_pause(app: &mut App, pause: PauseState) {
        app.world_mut()
            .resource_mut::<NextState<PauseState>>()
            .set(pause);
        app.update();
        app.update();
    }

    /// Pausing and resuming must NOT restart the game or leak gameplay entities.
    /// Pause is a `PauseState` sub-state of `Playing`, so toggling it must not exit
    /// `Playing` — which would re-run `level_setup` and start a fresh game. This is
    /// the regression guard for that (it caught the old computed-state design).
    #[test]
    fn pause_resume_preserves_the_engine_session() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
            .init_state::<GameState>()
            // LevelPlugin registers the PauseState sub-state of GameState::Playing.
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(test_assets())
            .add_plugins(LevelPlugin);

        // Enter a gameplay session.
        set_state(&mut app, GameState::Playing);
        assert!(
            app.world().get_resource::<EngineState>().is_some(),
            "entering Playing should run level_setup"
        );
        assert_eq!(count_cameras(&mut app), 1, "exactly one gameplay camera");

        // Lock a piece so the board is non-empty — our "did the engine reset" probe.
        {
            let mut engine = app.world_mut().resource_mut::<EngineState>();
            engine.0.step(InputFrame::default()); // spawn the first piece
            engine.0.step(InputFrame {
                hard_drop: true,
                ..default()
            }); // drop + lock it
        }
        let cells_before = app
            .world()
            .resource::<EngineState>()
            .0
            .snapshot()
            .board_cells
            .len();
        assert!(cells_before > 0, "setup: a piece should be locked");

        // Pause, then resume — via the PauseState SUB-state, not a GameState
        // transition. The whole point of the fix: `Playing` never exits.
        set_pause(&mut app, PauseState::Paused);
        set_pause(&mut app, PauseState::Running);

        // The session must survive: same board (no new game), one camera (no leak).
        let cells_after = app
            .world()
            .resource::<EngineState>()
            .0
            .snapshot()
            .board_cells
            .len();
        assert_eq!(
            cells_after, cells_before,
            "pause/resume reset the engine — the new-game bug"
        );
        assert_eq!(
            count_cameras(&mut app),
            1,
            "pause/resume duplicated the gameplay camera — the lag bug"
        );
    }
}
