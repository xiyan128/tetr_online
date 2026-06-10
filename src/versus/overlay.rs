//! Versus overlays: the countdown, the pause screen, and the result banner.
//!
//! All three are screen-space UI scoped to their [`VersusPhase`], drawn over
//! the live board scene (the result banner deliberately leaves the final
//! boards visible under a dim scrim — reading the losing stack is part of the
//! result). Navigation reuses the shared `FocusList` idiom so these screens
//! handle exactly like every other menu in the game.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button, theme, title_text};
use crate::GameState;

use super::{MatchClock, MatchOutcome, Participant, Seat, SeatStats, VersusConfig, VersusPhase};

/// Countdown pacing: three number beats, then a shorter "GO!".
const NUMBER_BEAT_SECONDS: f32 = 0.7;
const GO_BEAT_SECONDS: f32 = 0.5;

pub struct VersusOverlayPlugin;

impl Plugin for VersusOverlayPlugin {
    fn build(&self, app: &mut App) {
        app
            // Countdown
            .add_systems(OnEnter(VersusPhase::Countdown), spawn_countdown)
            .add_systems(
                Update,
                tick_countdown.run_if(in_state(VersusPhase::Countdown)),
            )
            // Pause
            .add_systems(
                Update,
                pause_on_keybind.run_if(in_state(VersusPhase::Running)),
            )
            .add_systems(OnEnter(VersusPhase::Paused), spawn_pause_overlay)
            .add_systems(
                Update,
                (focus_navigation::<PauseRoot>, pause_menu_activate)
                    .chain()
                    .run_if(in_state(VersusPhase::Paused)),
            )
            // Result
            .add_systems(OnEnter(VersusPhase::Over), spawn_result_banner)
            .add_systems(
                Update,
                (focus_navigation::<ResultRoot>, result_menu_activate)
                    .chain()
                    .run_if(in_state(VersusPhase::Over)),
            )
            .add_systems(
                Update,
                apply_rematch.run_if(resource_exists::<RematchRequested>),
            );
    }
}

// ---------------------------------------------------------------------------
// Countdown
// ---------------------------------------------------------------------------

/// The big center text; also carries the countdown clock.
#[derive(Component)]
struct CountdownText {
    elapsed: f32,
}

/// A full-screen, click-through column that centers its children.
fn overlay_root(scrim_alpha: f32) -> impl Bundle {
    (
        Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            row_gap: px(10),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, scrim_alpha)),
    )
}

fn spawn_countdown(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((
        overlay_root(0.0), // no scrim: the boards stay bright behind the count
        DespawnOnExit(VersusPhase::Countdown),
        children![(
            CountdownText { elapsed: 0.0 },
            Text::new("3"),
            TextFont {
                font: assets.font.clone(),
                font_size: 96.0,
                ..default()
            },
            TextColor(theme::ACCENT),
        )],
    ));
}

/// Advance the 3-2-1-GO beats; hand the match to `Running` when they finish.
/// Engines hold during the countdown (the step is `Running`-gated), so both
/// first pieces spawn on the same slice after "GO!".
fn tick_countdown(
    time: Res<Time>,
    text: Single<(&mut CountdownText, &mut Text)>,
    mut next: ResMut<NextState<VersusPhase>>,
) {
    let (mut state, mut text) = text.into_inner();
    state.elapsed += time.delta_secs();
    let total = 3.0 * NUMBER_BEAT_SECONDS + GO_BEAT_SECONDS;
    if state.elapsed >= total {
        next.set(VersusPhase::Running);
        return;
    }
    let label = match (state.elapsed / NUMBER_BEAT_SECONDS) as u32 {
        0 => "3",
        1 => "2",
        2 => "1",
        _ => "GO!",
    };
    if text.0 != label {
        text.0 = label.to_string();
    }
}

// ---------------------------------------------------------------------------
// Pause
// ---------------------------------------------------------------------------

#[derive(Component)]
struct PauseRoot;

#[derive(Component, Clone, Copy)]
enum PauseAction {
    Resume,
    Quit,
}

/// The player's pause keybind (Escape by default) freezes the whole match —
/// pausing a local match is inherently mutual. Works in bot-vs-bot too.
fn pause_on_keybind(
    keys: Res<ButtonInput<KeyCode>>,
    settings: Res<crate::settings::GameSettings>,
    mut next: ResMut<NextState<VersusPhase>>,
) {
    let (primary, secondary) = settings.keybinds.pause;
    let pressed = keys.just_pressed(primary) || secondary.is_some_and(|key| keys.just_pressed(key));
    if pressed {
        next.set(VersusPhase::Paused);
    }
}

fn spawn_pause_overlay(mut commands: Commands, assets: Res<GameAssets>) {
    let root = commands
        .spawn((
            PauseRoot,
            FocusList::new(2),
            overlay_root(0.55),
            DespawnOnExit(VersusPhase::Paused),
            children![title_text("PAUSED", assets.font.clone())],
        ))
        .id();
    let resume = commands
        .spawn((
            menu_button(0, "Resume", assets.font.clone()),
            PauseAction::Resume,
        ))
        .id();
    let quit = commands
        .spawn((
            menu_button(1, "Quit to Menu", assets.font.clone()),
            PauseAction::Quit,
        ))
        .id();
    let hint = commands
        .spawn(label_text("Esc to resume", assets.font.clone()))
        .id();
    commands.entity(root).add_children(&[resume, quit, hint]);
}

fn pause_menu_activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<PauseRoot>>,
    actions: Query<(&Focusable, &PauseAction)>,
    clicks: Query<(&Focusable, &Interaction)>,
    mut next_phase: ResMut<NextState<VersusPhase>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    let nav =
        read_nav_action(&keys, *list).or_else(|| clicked_focusable(&clicks).map(NavAction::Select));
    match nav {
        // Esc toggles straight back into the match.
        Some(NavAction::Back) => next_phase.set(VersusPhase::Running),
        Some(NavAction::Select(index)) => {
            for (focusable, action) in &actions {
                if focusable.index != index {
                    continue;
                }
                match action {
                    PauseAction::Resume => next_phase.set(VersusPhase::Running),
                    PauseAction::Quit => next_state.set(GameState::MainMenu),
                }
            }
        }
        None => {}
    }
}

// ---------------------------------------------------------------------------
// Result banner
// ---------------------------------------------------------------------------

#[derive(Component)]
struct ResultRoot;

#[derive(Component, Clone, Copy)]
enum ResultAction {
    Rematch,
    Menu,
}

/// Set by the result menu; consumed by [`apply_rematch`] (an exclusive system
/// — `restart_match` rebuilds the seats in place). `pub(crate)` so the match
/// tests can drive the exact path the Rematch button takes.
#[derive(Resource)]
pub(crate) struct RematchRequested;

/// The seat's display name, as the HUD labels it.
fn seat_label(config: &VersusConfig, registry: &crate::ai::ModelRegistry, seat: usize) -> String {
    match config.seats[seat] {
        Participant::Human => "YOU".to_string(),
        Participant::Bot { model } => registry.label(model).to_uppercase(),
    }
}

fn spawn_result_banner(
    mut commands: Commands,
    assets: Res<GameAssets>,
    outcome: Option<Res<MatchOutcome>>,
    config: Res<VersusConfig>,
    registry: Res<crate::ai::ModelRegistry>,
    clock: Res<MatchClock>,
    seats: Query<(&Seat, &SeatStats)>,
) {
    // The banner reads the world it was raised over; a missing outcome (manual
    // state poke in a test) reads as a draw rather than a panic.
    let winner = outcome.map(|o| o.winner).unwrap_or(None);
    let title = match winner {
        None => "DRAW".to_string(),
        Some(seat) => match config.seats[seat] {
            Participant::Human => "YOU WIN!".to_string(),
            Participant::Bot { .. } => {
                // A bot won. Against a human that reads as a loss; in
                // bot-vs-bot, name the victor.
                if config.seats.contains(&Participant::Human) {
                    "YOU LOSE".to_string()
                } else {
                    format!("{} WINS", seat_label(&config, &registry, seat))
                }
            }
        },
    };

    let mut stats = [SeatStats::default(); 2];
    for (seat, stat) in &seats {
        if seat.index < 2 {
            stats[seat.index] = *stat;
        }
    }
    let minutes = (clock.0 / 60.0) as u32;
    let seconds = clock.0 % 60.0;
    let summary = format!(
        "{}  ATK {}   ·   {}  ATK {}   ·   TIME {}:{:04.1}",
        seat_label(&config, &registry, 0),
        stats[0].attack_sent,
        seat_label(&config, &registry, 1),
        stats[1].attack_sent,
        minutes,
        seconds,
    );

    let root = commands
        .spawn((
            ResultRoot,
            FocusList::new(2),
            overlay_root(0.55),
            DespawnOnExit(VersusPhase::Over),
            children![title_text(title, assets.font.clone())],
        ))
        .id();
    let summary_id = commands
        .spawn(label_text(summary, assets.font.clone()))
        .id();
    let rematch = commands
        .spawn((
            menu_button(0, "Rematch", assets.font.clone()),
            ResultAction::Rematch,
        ))
        .id();
    let menu = commands
        .spawn((
            menu_button(1, "Main Menu", assets.font.clone()),
            ResultAction::Menu,
        ))
        .id();
    commands
        .entity(root)
        .add_children(&[summary_id, rematch, menu]);
}

fn result_menu_activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<ResultRoot>>,
    actions: Query<(&Focusable, &ResultAction)>,
    clicks: Query<(&Focusable, &Interaction)>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<GameState>>,
) {
    let nav =
        read_nav_action(&keys, *list).or_else(|| clicked_focusable(&clicks).map(NavAction::Select));
    match nav {
        Some(NavAction::Back) => next_state.set(GameState::MainMenu),
        Some(NavAction::Select(index)) => {
            for (focusable, action) in &actions {
                if focusable.index != index {
                    continue;
                }
                match action {
                    ResultAction::Rematch => commands.insert_resource(RematchRequested),
                    ResultAction::Menu => next_state.set(GameState::MainMenu),
                }
            }
        }
        None => {}
    }
}

/// Rebuild the match in place (fresh seed, fresh engines, back to the
/// countdown). Exclusive: `restart_match` reseats the world directly.
fn apply_rematch(world: &mut World) {
    world.remove_resource::<RematchRequested>();
    super::restart_match(world);
}
