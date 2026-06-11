//! Session setup: who sits at each board.
//!
//! Seat rows plus Start, in the shared `FocusList` idiom; [`SetupKind`] picks
//! the face. **Versus**: two rows (P1 offers "You" plus every registry model,
//! P2 the models — one keyboard, so exactly one human seat; a bot P1 gives
//! bot-vs-bot) and Start launches the match. **Watch AI**: one bot row, and
//! Start continues to mode select (the bot then plays the chosen variant on
//! one seat). A row cycles with Left/Right (Enter and click also cycle). The
//! selection writes [`SessionConfig`], which the session reads once on spawn;
//! the resource persists, so the screen remembers the last choice.

use bevy::prelude::*;

use crate::ai::ModelRegistry;
use crate::assets::GameAssets;
use crate::session::{Participant, SessionConfig};
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button_sized, screen_root, title_text};
use crate::GameState;

/// Which face the setup screen wears (set by the main menu before entering).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Default)]
pub enum SetupKind {
    #[default]
    Versus,
    WatchAi,
}

pub struct VersusSetupPlugin;

impl Plugin for VersusSetupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SetupKind>()
            .add_systems(OnEnter(GameState::SessionSetup), setup)
            .add_systems(
                Update,
                (
                    focus_navigation::<VersusSetupRoot>,
                    cycle_participants,
                    refresh_row_labels,
                    activate,
                )
                    .chain()
                    .run_if(in_state(GameState::SessionSetup)),
            );
    }
}

#[derive(Component)]
struct VersusSetupRoot;

/// Row 0 or 1: configures `SessionConfig.seats[seat]`.
#[derive(Component, Clone, Copy)]
struct SeatRow {
    seat: usize,
}

/// Row 2: starts the match.
#[derive(Component)]
struct StartRow;

/// The participants a seat row cycles through, in display order. Watch-AI
/// rows are bot-only (the whole point is watching one).
fn options_for(kind: SetupKind, seat: usize, registry: &ModelRegistry) -> Vec<Participant> {
    let mut options = Vec::new();
    if kind == SetupKind::Versus && seat == 0 {
        options.push(Participant::Human);
    }
    for model in 0..registry.len() {
        options.push(Participant::Bot { model });
    }
    options
}

fn participant_label(participant: Participant, registry: &ModelRegistry) -> String {
    match participant {
        Participant::Human => "You".to_string(),
        Participant::Bot { model } => registry.label(model).to_string(),
    }
}

fn setup(
    mut commands: Commands,
    assets: Res<GameAssets>,
    kind: Res<SetupKind>,
    registry: Res<ModelRegistry>,
    mut config: ResMut<SessionConfig>,
) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::SessionSetup)));
    let (title, seat_rows) = match *kind {
        SetupKind::Versus => ("VERSUS", 2),
        SetupKind::WatchAi => ("WATCH AI", 1),
    };
    // A Watch-AI visit must find a bot on seat 0 even if Play seated a human
    // there earlier; snap to the registry's first entry.
    if *kind == SetupKind::WatchAi && config.seats[0] == Participant::Human {
        config.seats[0] = Participant::Bot { model: 0 };
    }
    let _ = &registry; // (options are derived per-row below)

    let root = commands
        .spawn((
            VersusSetupRoot,
            FocusList::new(seat_rows + 1),
            screen_root(),
            DespawnOnExit(GameState::SessionSetup),
            children![title_text(title, assets.font.clone())],
        ))
        .id();

    // Seat rows are wide: "P2  < Best-First Attack >" needs the room.
    for seat in 0..seat_rows {
        let row = commands
            .spawn((
                menu_button_sized(seat, "", assets.font.clone(), 460.0),
                SeatRow { seat },
            ))
            .id();
        commands.entity(root).add_child(row);
    }
    let start = commands
        .spawn((
            menu_button_sized(seat_rows, "Start", assets.font.clone(), 220.0),
            StartRow,
        ))
        .id();
    let hint = commands
        .spawn(label_text(
            "Left/Right change seat  -  Enter start  -  Esc back",
            assets.font.clone(),
        ))
        .id();
    commands.entity(root).add_children(&[start, hint]);
}

/// Left/Right on a focused seat row cycles its participant.
fn cycle_participants(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<VersusSetupRoot>>,
    rows: Query<(&Focusable, &SeatRow)>,
    registry: Res<ModelRegistry>,
    kind: Res<SetupKind>,
    mut config: ResMut<SessionConfig>,
) {
    let step: isize = if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::KeyD)
    {
        1
    } else if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::KeyA) {
        -1
    } else {
        return;
    };
    let Some((_, row)) = rows.iter().find(|(f, _)| f.index == list.index) else {
        return;
    };
    cycle_seat(&mut config, &registry, *kind, row.seat, step);
}

/// Advance `seats[seat]` by `step` through its option ring.
fn cycle_seat(
    config: &mut SessionConfig,
    registry: &ModelRegistry,
    kind: SetupKind,
    seat: usize,
    step: isize,
) {
    let options = options_for(kind, seat, registry);
    let current = options
        .iter()
        .position(|p| *p == config.seats[seat])
        .unwrap_or(0);
    let next = (current as isize + step).rem_euclid(options.len() as isize) as usize;
    config.seats[seat] = options[next];
}

/// Keep the seat-row labels mirroring the config (initial fill included —
/// rows spawn with empty labels and this runs the same frame).
fn refresh_row_labels(
    config: Res<SessionConfig>,
    registry: Res<ModelRegistry>,
    kind: Res<SetupKind>,
    rows: Query<(&SeatRow, &Children)>,
    mut texts: Query<&mut Text>,
) {
    for (row, children) in &rows {
        let prefix = match *kind {
            SetupKind::Versus => format!("P{}", row.seat + 1),
            SetupKind::WatchAi => "BOT".to_string(),
        };
        let label = format!(
            "{}  < {} >",
            prefix,
            participant_label(config.seats[row.seat], &registry)
        );
        for child in children.iter() {
            if let Ok(mut text) = texts.get_mut(child) {
                if text.0 != label {
                    text.0 = label.clone();
                }
            }
        }
    }
}

/// Enter on Start begins the match; Enter/click on a seat row cycles it (so a
/// mouse-only player can configure everything); Esc backs out.
fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<VersusSetupRoot>>,
    rows: Query<(&Focusable, Option<&SeatRow>, Has<StartRow>)>,
    clicks: Query<(&Focusable, &Interaction), Changed<Interaction>>,
    registry: Res<ModelRegistry>,
    kind: Res<SetupKind>,
    mut config: ResMut<SessionConfig>,
    mut next: ResMut<NextState<GameState>>,
) {
    let nav =
        read_nav_action(&keys, *list).or_else(|| clicked_focusable(&clicks).map(NavAction::Select));
    match nav {
        Some(NavAction::Back) => next.set(GameState::MainMenu),
        Some(NavAction::Select(index)) => {
            for (focusable, seat_row, is_start) in &rows {
                if focusable.index != index {
                    continue;
                }
                if is_start {
                    match *kind {
                        SetupKind::Versus => {
                            info!(
                                "versus setup: starting {:?} vs {:?}",
                                config.seats[0], config.seats[1]
                            );
                            config.mode = crate::session::SessionMode::Versus;
                            next.set(GameState::Session);
                        }
                        // Watch-AI continues to mode select; the variant pick
                        // writes Solo{variant} and launches.
                        SetupKind::WatchAi => next.set(GameState::ModeSelect),
                    }
                } else if let Some(row) = seat_row {
                    cycle_seat(&mut config, &registry, *kind, row.seat, 1);
                }
            }
        }
        None => {}
    }
}
