use bevy::prelude::*;

use crate::engine::{Board, Cell, CellKind, Engine, EngineEvent, GameOverStatus};
use common::*;
use engine_bridge::*;

use crate::assets::GameAssets;
use crate::level::game_over::GameOverPlugin;
use crate::level::score::ScorePlugin;
use crate::level::sound_effects::SoundEffectsPlugin;
use crate::level::ui::UIPlugin;
use crate::player::{KeyboardController, KeyboardInput, PlayerController};
use crate::GameState;

mod common;
mod engine_bridge;
mod game_over;
mod score;
mod sound_effects;
mod ui;

pub struct LevelPlugin;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            // states
            .add_sub_state::<PlayingState>()
            // plugins
            .add_plugins(GameOverPlugin)
            .add_plugins(SoundEffectsPlugin)
            .add_plugins(ScorePlugin)
            .add_plugins(UIPlugin)
            // resources
            .init_resource::<LevelConfig>()
            .init_resource::<SimClock>()
            .init_resource::<FrameEvents>()
            // setup
            .add_systems(OnEnter(GameState::InGame), level_setup)
            // The driver runs first, then everything that reads its snapshot/events.
            .configure_sets(
                Update,
                (LevelSystems::EngineDriver, LevelSystems::Reconcile)
                    .chain()
                    .run_if(in_state(GameState::InGame)),
            )
            .add_systems(
                Update,
                engine_driver.in_set(LevelSystems::EngineDriver),
            )
            .add_systems(
                Update,
                (
                    update_playing_state,
                    reconcile_board_blocks,
                    reconcile_active_piece,
                    reconcile_ghost_piece,
                    emit_audio_cues,
                    handle_game_over,
                )
                    .in_set(LevelSystems::Reconcile),
            );
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Build the authoritative engine, the player controller, the background grid,
/// and the camera. Runs on entering `InGame` (including restart from game over).
fn level_setup(mut commands: Commands, config: Res<LevelConfig>, texture_assets: Res<GameAssets>) {
    info!("level_setup");

    let engine_config = engine_config_from_level(&config);
    let engine = Engine::new(engine_config, DEFAULT_SEED);
    let snapshot = engine.snapshot();

    // Seed the per-frame resources with a fresh engine + its initial snapshot so
    // the very first reconcile sees a consistent (empty) world.
    commands.insert_resource(EngineState(engine));
    commands.insert_resource(LatestSnapshot(snapshot));
    commands.insert_resource(FrameEvents::default());
    commands.insert_resource(SimClock::default());
    commands.insert_resource(PlayerInput(KeyboardController::new(das_config_from_level(
        &config,
    ))));

    // Background grid: a 10×height static field of dark cells, drawn once. This
    // is purely decorative scaffolding; gameplay minos are reconciled on top.
    let board = Board::with_top_margin(config.board_width, config.board_height, 20);
    let field = commands
        .spawn((
            GameField,
            Transform::default(),
            Visibility::default(),
            DespawnOnExit(GameState::InGame),
        ))
        .id();
    let mut block_ids = Vec::new();
    for (x, y) in board.coords() {
        let cell = Cell::new(x, y, CellKind::None);
        block_ids.push(spawn_free_block(
            &mut commands,
            &config,
            &texture_assets,
            &cell,
            BlockKind::Background,
        ));
    }
    commands.entity(field).add_children(&block_ids);

    // Camera centered on the visible board.
    commands.spawn((
        Camera2d,
        Transform::from_translation(Vec3::new(
            config.block_size * config.board_width as f32 / 2.,
            config.block_size * config.board_height as f32 / 2.,
            1.0,
        )),
        DespawnOnExit(GameState::InGame),
    ));
}

// ---------------------------------------------------------------------------
// Driver: input -> fixed-timestep engine steps -> snapshot + events
// ---------------------------------------------------------------------------

/// Accumulate real frame time and step the engine at a fixed sim rate.
///
/// Each fixed slice: poll the player controller against the *current* snapshot,
/// step the engine with `dt_seconds = SIM_DT_SECONDS`, and collect the events.
/// The controller is fed raw keyboard state once per frame (DAS still advances
/// per fixed slice via the staged input's `dt`). The latest snapshot and the
/// frame's events are published into resources for the reconcilers.
fn engine_driver(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut engine: ResMut<EngineState>,
    mut snapshot: ResMut<LatestSnapshot>,
    mut frame_events: ResMut<FrameEvents>,
    mut clock: ResMut<SimClock>,
    mut player: ResMut<PlayerInput>,
) {
    frame_events.0.clear();

    clock.accumulator_seconds += time.delta_secs();
    // Guard against spiral-of-death after a long stall (e.g. tab backgrounded).
    let max_accumulated = SIM_DT_SECONDS * 8.0;
    if clock.accumulator_seconds > max_accumulated {
        clock.accumulator_seconds = max_accumulated;
    }

    let mut stepped = false;
    while clock.accumulator_seconds >= SIM_DT_SECONDS {
        clock.accumulator_seconds -= SIM_DT_SECONDS;

        // Stage this fixed slice's input. Edge-triggered actions (rotate, hold,
        // hard drop) are only honored on the first slice of the frame so one key
        // press maps to one action even if several slices run this frame.
        let input = KeyboardInput::from_keyboard(&keyboard, SIM_DT_SECONDS);
        let input = if stepped { suppress_edges(input) } else { input };
        player.0.set_input(input);

        let frame = player.0.poll(&snapshot.0);
        let events = engine.0.step(frame);
        snapshot.0 = engine.0.snapshot();
        frame_events.0.extend(events);
        stepped = true;
    }

    // If no slice ran this frame the snapshot is unchanged; reconcilers reading
    // it are idempotent, so nothing to do.
}

/// Clear edge-triggered (just-pressed) flags so repeated fixed slices in one
/// frame don't replay a single key press. Held flags (`soft_drop`, the
/// `*_pressed` used by DAS) are preserved.
fn suppress_edges(mut input: KeyboardInput) -> KeyboardInput {
    input.left_just_pressed = false;
    input.right_just_pressed = false;
    input.hard_drop_just_pressed = false;
    input.rotate_cw_just_pressed = false;
    input.rotate_ccw_just_pressed = false;
    input.hold_just_pressed = false;
    input.pause_just_pressed = false;
    input
}

// ---------------------------------------------------------------------------
// Reconcilers: render entities from the snapshot
// ---------------------------------------------------------------------------

/// Derive Falling vs. Locking from the snapshot's active piece. The lock-down
/// timer bar's UI keys off this sub-state.
fn update_playing_state(
    snapshot: Res<LatestSnapshot>,
    current: Res<State<PlayingState>>,
    mut next: ResMut<NextState<PlayingState>>,
) {
    let landed = snapshot
        .0
        .active
        .as_ref()
        .map(|active| active.landed)
        .unwrap_or(false);

    let desired = if landed {
        PlayingState::Locking
    } else {
        PlayingState::Falling
    };
    if current.get() != &desired {
        next.set(desired);
    }
}

/// Rebuild the locked-board minos whenever they change. Board state only
/// changes on a lock/clear, so this is cheap: we cache the last board cells and
/// despawn+respawn only on a diff.
fn reconcile_board_blocks(
    mut commands: Commands,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    existing: Query<Entity, With<StaticBlock>>,
    mut last: Local<Option<Vec<crate::engine::SnapshotCell>>>,
) {
    let cells = &snapshot.0.board_cells;
    if last.as_ref() == Some(cells) {
        return;
    }

    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }
    for cell in cells {
        spawn_snapshot_block(
            &mut commands,
            &config,
            &texture_assets,
            cell.x,
            cell.y,
            cell.piece_type,
            BlockKind::Static,
        );
    }

    *last = Some(cells.clone());
}

/// Rebuild the active-piece minos each frame from `snapshot.active`. Despawn all
/// and respawn from the snapshot's absolute cell coords — cheap (4 sprites) and
/// always in sync with the engine.
fn reconcile_active_piece(
    mut commands: Commands,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    existing: Query<Entity, With<FallingBlock>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }
    let Some(active) = snapshot.0.active.as_ref() else {
        return;
    };
    for cell in &active.cells {
        spawn_snapshot_block(
            &mut commands,
            &config,
            &texture_assets,
            cell.x,
            cell.y,
            cell.piece_type,
            BlockKind::Falling,
        );
    }
}

/// Rebuild the ghost-piece minos each frame from `snapshot.ghost_cells`.
/// Reconciled after the active piece so both read the same snapshot and the
/// ghost can never lag the piece.
fn reconcile_ghost_piece(
    mut commands: Commands,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    existing: Query<Entity, With<GhostBlock>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }
    // Hide the ghost when it would coincide with the active piece (piece already
    // resting on the stack), matching the old renderer's "no ghost when grounded".
    let landed = snapshot
        .0
        .active
        .as_ref()
        .map(|active| active.landed)
        .unwrap_or(true);
    if landed {
        return;
    }
    for cell in &snapshot.0.ghost_cells {
        spawn_snapshot_block(
            &mut commands,
            &config,
            &texture_assets,
            cell.x,
            cell.y,
            cell.piece_type,
            BlockKind::Ghost,
        );
    }
}

// ---------------------------------------------------------------------------
// Events: audio + game over
// ---------------------------------------------------------------------------

/// Map this frame's engine events onto [`AudioCue`]s, preserving the same SFX
/// moments as the pre-migration renderer:
///   * Rotated   -> Rotation
///   * HardDropped -> HardDrop
///   * Held      -> Hold
///   * a piece grounding (Locking transition) -> Placed
///   * Locked(n) -> Locked(n) (lock thunk / line-clear jingles)
fn emit_audio_cues(
    mut commands: Commands,
    frame_events: Res<FrameEvents>,
    mut prev_landed: Local<bool>,
    snapshot: Res<LatestSnapshot>,
) {
    for event in &frame_events.0 {
        match event {
            EngineEvent::Rotated { .. } => commands.trigger(AudioCue::Rotation),
            EngineEvent::HardDropped { .. } => commands.trigger(AudioCue::HardDrop),
            EngineEvent::Held { .. } => commands.trigger(AudioCue::Hold),
            EngineEvent::Locked { lines_cleared, .. } => {
                commands.trigger(AudioCue::Locked(*lines_cleared))
            }
            _ => {}
        }
    }

    // "Placed" plays on the rising edge of the active piece becoming grounded
    // (the moment the lock-down timer starts), mirroring the old detect_placement.
    let landed = snapshot
        .0
        .active
        .as_ref()
        .map(|active| active.landed)
        .unwrap_or(false);
    if landed && !*prev_landed {
        commands.trigger(AudioCue::Placed);
    }
    *prev_landed = landed;
}

/// Transition to the game-over screen when the engine reports it. Reads the
/// snapshot (authoritative) rather than racing the event list.
fn handle_game_over(
    snapshot: Res<LatestSnapshot>,
    mut next_game_state: ResMut<NextState<GameState>>,
) {
    if let Some(reason) = snapshot.0.game_over {
        match reason {
            GameOverStatus::BlockOut => info!("game over: block out"),
            GameOverStatus::LockOut => info!("game over: lock out"),
        }
        next_game_state.set(GameState::GameOver);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, InputFrame};

    #[test]
    fn level_plugin_systems_initialize_without_query_conflicts() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
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
            .add_plugins(LevelPlugin);

        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::InGame);
        app.update();
        // A second update applies the InGame state transition and runs setup.
        app.update();

        assert!(app.world().get_resource::<EngineState>().is_some());
        assert!(app.world().get_resource::<LatestSnapshot>().is_some());

        // Force the fixed-timestep driver to run a slice (independent of the test
        // harness's wall clock) so the engine spawns its first piece, then verify
        // the active-piece reconciler materialized it into FallingBlock entities.
        app.world_mut()
            .resource_mut::<SimClock>()
            .accumulator_seconds = SIM_DT_SECONDS;
        app.update();

        let snapshot = &app.world().resource::<LatestSnapshot>().0;
        assert!(
            snapshot.active.is_some(),
            "driver should have stepped the engine and spawned a piece"
        );
        let falling_blocks = app
            .world_mut()
            .query_filtered::<(), With<FallingBlock>>()
            .iter(app.world())
            .count();
        assert_eq!(
            falling_blocks, 4,
            "reconcile_active_piece must render the active piece's 4 cells"
        );
    }

    /// The driver accumulator must advance the engine deterministically: feeding
    /// a fixed total of wall-clock dt produces the same engine state as stepping
    /// the engine directly with the same fixed slices. This pins the
    /// fixed-timestep contract (gravity/lock-down advance independent of frame
    /// rate) without booting a render App.
    #[test]
    fn driver_accumulator_advances_engine_deterministically() {
        // Reference engine: step exactly N fixed slices with no input.
        let config = EngineConfig::default();
        let mut reference = Engine::new(config.clone(), DEFAULT_SEED);
        let slices = 10;
        for _ in 0..slices {
            reference.step(InputFrame {
                dt_seconds: SIM_DT_SECONDS,
                ..InputFrame::default()
            });
        }

        // Driver-style: accumulate one frame whose dt equals N fixed slices, then
        // drain the accumulator in fixed slices (the engine_driver loop).
        let mut driven = Engine::new(config, DEFAULT_SEED);
        let mut accumulator = SIM_DT_SECONDS * slices as f32;
        let mut ran = 0;
        while accumulator >= SIM_DT_SECONDS {
            accumulator -= SIM_DT_SECONDS;
            driven.step(InputFrame {
                dt_seconds: SIM_DT_SECONDS,
                ..InputFrame::default()
            });
            ran += 1;
        }

        assert_eq!(ran, slices, "accumulator must drain into exactly N slices");
        assert_eq!(
            driven.snapshot(),
            reference.snapshot(),
            "fixed-slice driving matches direct fixed stepping"
        );
    }

    #[test]
    fn suppress_edges_keeps_held_flags_but_drops_just_pressed() {
        let input = KeyboardInput {
            dt_seconds: SIM_DT_SECONDS,
            left_pressed: true,
            left_just_pressed: true,
            soft_drop: true,
            hard_drop_just_pressed: true,
            rotate_cw_just_pressed: true,
            hold_just_pressed: true,
            ..KeyboardInput::default()
        };
        let suppressed = suppress_edges(input);

        assert!(suppressed.left_pressed, "held flags survive");
        assert!(suppressed.soft_drop, "soft drop is a held flag");
        assert!(!suppressed.left_just_pressed);
        assert!(!suppressed.hard_drop_just_pressed);
        assert!(!suppressed.rotate_cw_just_pressed);
        assert!(!suppressed.hold_just_pressed);
    }

    #[test]
    fn engine_config_bridge_maps_level_dimensions() {
        let level = LevelConfig::default();
        let engine = engine_config_from_level(&level);

        assert_eq!(engine.board_width, level.board_width);
        assert_eq!(engine.visible_height, level.board_height);
        assert_eq!(engine.preview_count, level.preview_count);
        // The engine does not carry DAS; that lives in DasConfig.
        let das = das_config_from_level(&level);
        assert_eq!(das.delay_seconds, level.das_delay.as_secs_f32());
        assert_eq!(das.repeat_seconds, level.das_repeat_duration.as_secs_f32());
    }

    fn headless_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
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
            .add_plugins(LevelPlugin);
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::InGame);
        app.update();
        app.update();
        app
    }

    /// Drive a hard drop through the full app and verify the board reconciler
    /// mirrors the engine's locked board into StaticBlock entities. This exercises
    /// the snapshot -> board reconciliation path end to end.
    #[test]
    fn hard_drop_reconciles_locked_board_into_static_blocks() {
        let mut app = headless_app();

        // Step once to spawn the first piece.
        app.world_mut()
            .resource_mut::<SimClock>()
            .accumulator_seconds = SIM_DT_SECONDS;
        app.update();

        // Press Space (hard drop) and force another driver slice.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Space);
        app.world_mut()
            .resource_mut::<SimClock>()
            .accumulator_seconds = SIM_DT_SECONDS;
        app.update();

        let snapshot = &app.world().resource::<LatestSnapshot>().0;
        assert_eq!(
            snapshot.board_cells.len(),
            4,
            "hard drop should lock the piece's 4 cells onto the board"
        );
        let static_blocks = app
            .world_mut()
            .query_filtered::<(), With<StaticBlock>>()
            .iter(app.world())
            .count();
        assert_eq!(
            static_blocks, 4,
            "reconcile_board_blocks must mirror the locked board cells"
        );
    }
}
