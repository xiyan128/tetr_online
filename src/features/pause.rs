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
//! ## Cross-cutting contract this relies on
//! Gameplay entities and the level `OnEnter` setup must be scoped to the *whole
//! session* (Playing **or** Paused), not to `Playing` alone. Otherwise the
//! `Playing -> Paused` transition would fire `DespawnOnExit(GameState::Playing)`
//! (despawning the board) and `Paused -> Playing` would re-run `level_setup`
//! (rebuilding a fresh `Engine`), resetting the game. The integrator wires a
//! `InGameplay` computed state for that scoping; see this module's report. The
//! per-frame driver/reconciler/UI-update run conditions stay on `Playing` so the
//! sim still freezes while paused. This file does not depend on the computed
//! state by name — it only hides/restores visibility and toggles `GameState`.
//!
//! Touch only this file.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::level::common::{BackgroundBlock, FallingBlock, GhostBlock, PreviewBlock, StaticBlock};
use crate::settings::{GameAction, GameSettings};
use crate::ui::focus::{focus_navigation, read_nav_action, FocusList, Focusable, NavAction};
use crate::ui::widgets::{label_text, menu_button, screen_root, title_text};
use crate::{GameState, InGameplay};

/// Pause overlay + `Playing <-> Paused` toggle.
pub struct PausePlugin;

impl Plugin for PausePlugin {
    fn build(&self, app: &mut App) {
        app
            // Enter pause from gameplay on the Pause keybind.
            .add_systems(
                Update,
                toggle_to_paused.run_if(in_state(GameState::Playing)),
            )
            // Build the overlay on pause; DespawnOnExit(Paused) tears it down.
            .add_systems(OnEnter(GameState::Paused), setup_pause_overlay)
            // Conceal the playfield iff paused, re-evaluated EVERY frame. Doing
            // this idempotently (instead of a one-shot OnEnter-hide / OnExit-show
            // pair) means rapid Esc toggling can never leave the board stuck
            // hidden or stuck shown — visibility always matches the live state.
            .add_systems(
                Update,
                sync_gameplay_visibility.run_if(in_state(InGameplay)),
            )
            // Drive the overlay's focus + selection while paused.
            .add_systems(
                Update,
                (focus_navigation::<PauseRoot>, activate)
                    .chain()
                    .run_if(in_state(GameState::Paused)),
            );
    }
}

/// Menu rows on the pause overlay, in display (focus) order.
#[derive(Component, Clone, Copy)]
enum PauseAction {
    Resume,
    Quit,
}

const ITEMS: [(PauseAction, &str); 2] = [
    (PauseAction::Resume, "Resume"),
    (PauseAction::Quit, "Quit to Menu"),
];

/// Marker for the pause overlay's screen-root (carries the [`FocusList`]).
#[derive(Component)]
struct PauseRoot;

/// True on the rising edge of the bound Pause action (primary or secondary key).
fn pause_just_pressed(keys: &ButtonInput<KeyCode>, settings: &GameSettings) -> bool {
    let (primary, secondary) = settings.keybinds.get(GameAction::Pause);
    keys.just_pressed(primary) || secondary.is_some_and(|key| keys.just_pressed(key))
}

/// While playing, the Pause keybind transitions to [`GameState::Paused`].
fn toggle_to_paused(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<GameSettings>,
    mut next: ResMut<NextState<GameState>>,
) {
    if pause_just_pressed(&keys, &settings) {
        next.set(GameState::Paused);
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
            DespawnOnExit(GameState::Paused),
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
    mut next: ResMut<NextState<GameState>>,
) {
    // The Pause keybind (when it isn't Escape) also resumes, mirroring the
    // Playing-side toggle so the same key pauses and unpauses.
    let (primary, _) = settings.keybinds.get(GameAction::Pause);
    if primary != KeyCode::Escape && pause_just_pressed(&keys, &settings) {
        next.set(GameState::Playing);
        return;
    }

    // `iter().next()` rather than `single()`: during fast toggling there can be a
    // transient frame with zero or more-than-one overlay root, and we must not
    // get stuck unable to resume in that frame.
    let Some(list) = lists.iter().next() else {
        return;
    };
    match read_nav_action(&keys, list) {
        Some(NavAction::Back) => next.set(GameState::Playing),
        Some(NavAction::Select(index)) => {
            for (focusable, action) in &actions {
                if focusable.index != index {
                    continue;
                }
                match action {
                    PauseAction::Resume => next.set(GameState::Playing),
                    PauseAction::Quit => next.set(GameState::MainMenu),
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
    state: Res<State<GameState>>,
    mut content: Query<&mut Visibility, GameplayContent>,
) {
    let desired = if *state.get() == GameState::Paused {
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
