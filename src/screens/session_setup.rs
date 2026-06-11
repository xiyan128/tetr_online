//! Versus setup: who sits at each board.
//!
//! Two seat rows plus Start, in the shared `FocusList` idiom. A seat row
//! cycles its participant with Left/Right (Enter and click also cycle, so
//! every input path works); P1 offers "You" plus every registry model, P2
//! offers the models (one keyboard, so exactly one human seat in v1 — making
//! P1 a bot gives bot-vs-bot, the versus twin of Watch-AI). The selection
//! writes [`SessionConfig`], which the match reads once on spawn; the resource
//! persists, so the screen remembers the last matchup.

use bevy::prelude::*;

use crate::ai::ModelRegistry;
use crate::assets::GameAssets;
use crate::session::{Participant, SessionConfig};
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button_sized, screen_root, title_text};
use crate::GameState;

pub struct VersusSetupPlugin;

impl Plugin for VersusSetupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::SessionSetup), setup)
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

/// The participants a seat row cycles through, in display order.
fn options_for(seat: usize, registry: &ModelRegistry) -> Vec<Participant> {
    let mut options = Vec::new();
    if seat == 0 {
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

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::SessionSetup)));
    let root = commands
        .spawn((
            VersusSetupRoot,
            FocusList::new(3),
            screen_root(),
            DespawnOnExit(GameState::SessionSetup),
            children![title_text("VERSUS", assets.font.clone())],
        ))
        .id();

    // Seat rows are wide: "P2  < Best-First Attack >" needs the room.
    for seat in 0..2 {
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
            menu_button_sized(2, "Start", assets.font.clone(), 220.0),
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
    cycle_seat(&mut config, &registry, row.seat, step);
}

/// Advance `seats[seat]` by `step` through its option ring.
fn cycle_seat(config: &mut SessionConfig, registry: &ModelRegistry, seat: usize, step: isize) {
    let options = options_for(seat, registry);
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
    rows: Query<(&SeatRow, &Children)>,
    mut texts: Query<&mut Text>,
) {
    for (row, children) in &rows {
        let label = format!(
            "P{}  < {} >",
            row.seat + 1,
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
                    info!(
                        "versus setup: starting {:?} vs {:?}",
                        config.seats[0], config.seats[1]
                    );
                    next.set(GameState::Session);
                } else if let Some(row) = seat_row {
                    cycle_seat(&mut config, &registry, row.seat, 1);
                }
            }
        }
        None => {}
    }
}
