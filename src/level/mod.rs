//! The in-game level: wires the engine into the Bevy world.
//!
//! [`LevelPlugin`] owns the per-frame pipeline that drives the
//! [`Engine`](crate::engine::Engine): it samples and latches input in
//! `PreUpdate`, steps the engine once per fixed slice in `FixedUpdate`, and
//! reconciles the published snapshot into renderable/audio state in `Update`.
//! Submodules cover the engine bridge, score/sound/game-over reconcilers, the
//! in-game UI, and the shared [`common`] types. The plugin is written to run
//! standalone (e.g. in headless tests) by initialising its shared resources
//! idempotently rather than depending on `GamePlugin`.

use bevy::prelude::*;

use crate::engine::{Board, Cell, CellKind, Engine, EngineEvent, GameOverStatus};
use common::*;
use engine_bridge::*;

use crate::assets::GameAssets;
use crate::level::game_over::GameOverPlugin;
use crate::level::score::ScorePlugin;
use crate::level::sound_effects::SoundEffectsPlugin;
use crate::level::ui::UIPlugin;
use crate::player::{KeyboardController, PlayerController};
use crate::{GameState, PauseState};

pub(crate) mod common;
pub(crate) mod engine_bridge;
mod game_over;
mod score;
mod sound_effects;
mod ui;

// The engine-driver resources the per-frame pipeline owns. Re-exported so the AI
// sandbox driver (AI3.6) — which steps the *same* engine with a different
// controller — can read/publish them without reaching into `engine_bridge`.
pub use engine_bridge::{EngineState, FrameEvents, LatestSnapshot, PlayerInput, SIM_DT_SECONDS};

pub struct LevelPlugin;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            // states
            .add_sub_state::<PlayingState>()
            // Pause sub-state of `Playing` (Running/Paused). Modeled as a sub-state
            // rather than a sibling `GameState` so toggling pause never exits
            // `Playing` — the session (engine, board, camera, HUD; all scoped to
            // `OnEnter(GameState::Playing)`) survives a pause/resume round-trip
            // instead of being despawned and rebuilt (a `ComputedStates` re-runs
            // OnEnter/OnExit on every source change, which restarted the game).
            .add_sub_state::<PauseState>()
            // plugins
            .add_plugins(GameOverPlugin)
            .add_plugins(SoundEffectsPlugin)
            .add_plugins(ScorePlugin)
            .add_plugins(UIPlugin)
            // Fixed simulation clock. The engine steps in `FixedUpdate` (see
            // below), so Bevy's accumulator — not a hand-rolled one — decides how
            // many slices run per render frame. Seeded from the same `SIM_HZ` the
            // engine bridge exposes so the fixed `dt` equals `SIM_DT_SECONDS`.
            .insert_resource(Time::<Fixed>::from_hz(SIM_HZ as f64))
            // resources
            .init_resource::<LevelConfig>()
            .init_resource::<HeldInput>()
            .init_resource::<PendingEdges>()
            .init_resource::<FrameEvents>()
            // Reflection registration for inspector/scene support. Engine-wrapping
            // resources (EngineState/LatestSnapshot/FrameEvents/HeldInput/
            // PendingEdges) are deliberately NOT registered: reflecting them would
            // force Bevy `Reflect` onto the engine-agnostic crate.
            .register_type::<LevelConfig>()
            .register_type::<GameField>()
            .register_type::<BackgroundBlock>()
            .register_type::<FallingBlock>()
            .register_type::<StaticBlock>()
            .register_type::<GhostBlock>()
            .register_type::<PreviewBlock>()
            .register_type::<GameplayCamera>()
            // Shared M1 contracts the gameplay systems read. `init_resource` is
            // idempotent, so `GamePlugin` remains the canonical owner while
            // `LevelPlugin` stays self-sufficient (e.g. in headless tests).
            // Contract: keep `init_resource` here — do NOT switch to
            // `insert_resource`, which would clobber the values `GamePlugin`
            // already inserted (e.g. the player's persisted `GameSettings`).
            .init_resource::<crate::settings::GameSettings>()
            .init_resource::<crate::variant::ActiveVariant>()
            .init_resource::<crate::variant::VariantProgress>()
            // setup — scoped to the gameplay session (`GameState::Playing`). Pause
            // is a sub-state of `Playing`, so a pause/resume round-trip does NOT
            // re-enter `Playing`: this runs exactly once per game and the engine is
            // never rebuilt mid-session.
            .add_systems(
                OnEnter(GameState::Playing),
                (level_setup, crate::variant::reset_variant_progress),
            )
            // Per-frame pipeline across three schedules (Bevy runs them
            // First → PreUpdate → FixedMain → Update each frame):
            //
            //  * PreUpdate  — `LevelSystems::EngineDriver`: clear this frame's
            //    event buffer and sample/latch keyboard input. Runs *before*
            //    `FixedUpdate` so each slice sees fresh held flags + edges, and
            //    after `InputSystems` so `just_pressed` is still valid (Bevy
            //    clears it earlier in PreUpdate).
            //  * FixedUpdate — `step_engine`: drain the latch, step the engine
            //    once per accumulated slice, accumulate events.
            //  * Update     — `LevelSystems::Reconcile`: render/UI/audio systems
            //    read the published snapshot + the frame's accumulated events.
            //
            // The two `LevelSystems` sets keep `EngineDriver` "before" `Reconcile`
            // for external `.after(EngineDriver)` consumers (e.g. info_panel); the
            // schedule order already guarantees PreUpdate runs before Update. Both
            // gate on `PauseState::Running` (which implies `Playing`), so the whole
            // per-frame pipeline freezes while paused without despawning anything.
            .configure_sets(
                PreUpdate,
                LevelSystems::EngineDriver
                    .after(bevy::input::InputSystems)
                    .run_if(in_state(PauseState::Running)),
            )
            .configure_sets(
                Update,
                LevelSystems::Reconcile.run_if(in_state(PauseState::Running)),
            )
            .add_systems(
                PreUpdate,
                (clear_frame_events, latch_input)
                    .chain()
                    .in_set(LevelSystems::EngineDriver),
            )
            // Keyboard driver. Gated on the AI sandbox being *off* so it and the
            // AI driver (`crate::ai::sandbox::step_engine_ai`) are mutually
            // exclusive: a sandbox session is driven by the bot, a normal game by
            // the keyboard, and the keyboard path is byte-identical when the
            // sandbox is unused (the condition defaults to keyboard if the
            // `AiSandbox` flag resource is absent — e.g. headless level tests).
            .add_systems(
                FixedUpdate,
                step_engine.run_if(
                    in_state(PauseState::Running).and(not(crate::ai::sandbox::sandbox_active)),
                ),
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
                    crate::variant::check_variant_end_conditions,
                )
                    .in_set(LevelSystems::Reconcile),
            );
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

/// Build the authoritative engine, the player controller, the background grid,
/// and the camera. Runs on entering `Playing` (including restart from game over).
fn level_setup(
    mut commands: Commands,
    mut config: ResMut<LevelConfig>,
    settings: Res<crate::settings::GameSettings>,
    active_variant: Res<crate::variant::ActiveVariant>,
    texture_assets: Res<GameAssets>,
) {
    info!("level_setup ({})", active_variant.0.display_name());

    // Mirror the player's chosen next-count into LevelConfig so the previewer
    // (which reads LevelConfig.preview_count) and the engine queue agree.
    config.preview_count = settings.next_count;

    let engine_config = engine_config_for_game(&config, &settings, active_variant.0);
    let engine = Engine::new(engine_config, DEFAULT_SEED);
    let snapshot = engine.snapshot();

    // Seed the per-frame resources with a fresh engine + its initial snapshot so
    // the very first reconcile sees a consistent (empty) world.
    commands.insert_resource(EngineState(engine));
    commands.insert_resource(LatestSnapshot(snapshot));
    commands.insert_resource(FrameEvents::default());
    commands.insert_resource(HeldInput::default());
    commands.insert_resource(PendingEdges::default());
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
            DespawnOnExit(GameState::Playing),
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

    // Camera centered on the visible board. Tagged `GameplayCamera` so visual-FX
    // systems (shake/bloom/CRT) can target it without disturbing menu cameras.
    commands.spawn((
        Camera2d,
        GameplayCamera,
        Transform::from_translation(camera_center(&config)),
        DespawnOnExit(GameState::Playing),
    ));
}

// ---------------------------------------------------------------------------
// Driver: input (PreUpdate) -> fixed engine step (FixedUpdate) -> snapshot/events
// ---------------------------------------------------------------------------

/// Clear the per-frame engine-event buffer once, *before* the fixed slices run.
///
/// `FrameEvents` accumulates the events of every fixed slice that runs this
/// render frame, so the `Update` reconcilers see the full batch. It must be
/// cleared here in `PreUpdate` (before `FixedUpdate`) rather than inside the
/// step — clearing per slice would drop the events of earlier slices in the same
/// frame, and clearing in `Update` would wipe the batch before anything reads it.
fn clear_frame_events(mut frame_events: ResMut<FrameEvents>) {
    frame_events.0.clear();
}

/// Sample the keyboard once per render frame, latch its just-pressed edges, and
/// stage the held flags for the fixed step.
///
/// Runs in `PreUpdate` after `InputSystems`, where `just_pressed` is still valid
/// for this frame. Bevy clears `just_pressed` earlier in `PreUpdate` and the
/// engine steps in `FixedUpdate` (which runs zero-or-more times per frame), so
/// reading edges directly in the step would drop a press on a zero-slice frame
/// and duplicate it on a multi-slice frame. Latching here and draining once in
/// the step (then resetting) is the drop/dup-safe pattern.
///
/// Keybinds come from the player's settings so options-screen rebinds take effect
/// (defaults reproduce the legacy hard-coded mapping).
fn latch_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    settings: Res<crate::settings::GameSettings>,
    mut held: ResMut<HeldInput>,
    mut pending: ResMut<PendingEdges>,
) {
    let raw = crate::features::options::keyboard_input_from_keybinds(
        &keyboard,
        &settings.keybinds,
        settings.hold_enabled,
        SIM_DT_SECONDS,
    );
    pending.latch(&raw);
    // Stage this frame's held flags (DAS direction, soft drop) + per-slice dt for
    // the fixed step. Edge fields are unused by the step (edges come from the
    // latch) but ride along so dt + held travel as one value.
    held.0 = raw;
}

/// Step the engine once per fixed slice. Bevy's `FixedUpdate` runs this zero or
/// more times per render frame, draining `Time::<Fixed>`'s accumulator; the
/// spiral-of-death guard (long stalls) is Bevy's `Time<Virtual>::max_delta`, so
/// no hand-rolled accumulator is needed.
///
/// Each slice: build the [`RawKeyboardFrame`] from this frame's staged held flags
/// plus the latched edges, poll the controller against the *current* snapshot,
/// step the engine with `dt = time.delta_secs()` (equals [`SIM_DT_SECONDS`]),
/// publish the snapshot, and append the events. The latch is drained onto the
/// first slice and immediately [`reset`](PendingEdges::reset) so a press fires
/// exactly once even when several slices run in one frame; held flags persist so
/// DAS auto-repeat and soft drop keep advancing across slices.
fn step_engine(
    time: Res<Time<Fixed>>,
    held: Res<HeldInput>,
    mut engine: ResMut<EngineState>,
    mut snapshot: ResMut<LatestSnapshot>,
    mut frame_events: ResMut<FrameEvents>,
    mut player: ResMut<PlayerInput>,
    mut pending: ResMut<PendingEdges>,
) {
    let mut input = held.0;
    input.dt_seconds = time.delta_secs();
    pending.drain_onto(&mut input);
    player.0.set_input(input);

    let frame = player.0.poll(&snapshot.0);
    let events = engine.0.step(frame);
    snapshot.0 = engine.0.snapshot();
    frame_events.0.extend(events);

    pending.reset();
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
        .is_some_and(|active| active.landed);

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
    settings: Res<crate::settings::GameSettings>,
    texture_assets: Res<GameAssets>,
    existing: Query<Entity, With<GhostBlock>>,
) {
    for entity in existing.iter() {
        commands.entity(entity).despawn();
    }
    // Player can disable the ghost entirely (GameSettings.ghost_enabled).
    if !settings.ghost_enabled {
        return;
    }
    // Hide the ghost when it would coincide with the active piece (piece already
    // resting on the stack), matching the old renderer's "no ghost when grounded".
    // No active piece ⇒ treat as "landed" so the ghost stays hidden.
    let landed = snapshot
        .0
        .active
        .as_ref()
        .is_none_or(|active| active.landed);
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
        .is_some_and(|active| active.landed);
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
    use crate::engine::InputFrame;
    use bevy::time::TimeUpdateStrategy;
    use core::time::Duration;

    /// Run exactly `n` `FixedUpdate` slices through the *real* schedule.
    ///
    /// `TimeUpdateStrategy::FixedTimesteps(n)` makes Bevy advance virtual time by
    /// exactly `n` fixed periods per `App::update`, so the `RunFixedMainLoop`
    /// runner iterates `FixedMain` `n` times with `Time::<Fixed>::delta()` equal
    /// to the timestep each iteration — independent of the test harness's wall
    /// clock. This drives the production pipeline (PreUpdate latch → FixedUpdate
    /// step → Update reconcile) for `n` deterministic ticks.
    fn tick_fixed(app: &mut App, n: u32) {
        app.insert_resource(TimeUpdateStrategy::FixedTimesteps(n));
        app.update();
    }

    /// A headless `LevelPlugin` app pinned to a deterministic clock: setup runs
    /// with zero fixed slices (manual duration of zero) so the engine doesn't
    /// step until a test asks for ticks via [`tick_fixed`].
    fn headless_app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
            // Freeze the clock during setup; tests advance it explicitly. Without
            // this the default `Automatic` strategy would run wall-clock-dependent
            // fixed slices during the setup `update()`s below.
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
            .add_plugins(LevelPlugin);
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Playing);
        app.update();
        // A second update applies the Playing state transition and runs setup.
        app.update();
        app
    }

    #[test]
    fn level_plugin_systems_initialize_without_query_conflicts() {
        let mut app = headless_app();

        assert!(app.world().get_resource::<EngineState>().is_some());
        assert!(app.world().get_resource::<LatestSnapshot>().is_some());

        // Run one fixed slice through the real FixedUpdate schedule so the engine
        // spawns its first piece, then verify the active-piece reconciler (Update)
        // materialized it into FallingBlock entities.
        tick_fixed(&mut app, 1);

        let snapshot = &app.world().resource::<LatestSnapshot>().0;
        assert!(
            snapshot.active.is_some(),
            "the fixed step should have advanced the engine and spawned a piece"
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

    /// The `FixedUpdate` step must advance the engine deterministically: running N
    /// fixed slices through the real schedule produces the same engine state as
    /// stepping the engine directly with N fixed slices of `SIM_DT_SECONDS`. This
    /// pins the fixed-timestep contract (gravity/lock-down advance independent of
    /// render frame rate) now that Bevy's `Time<Fixed>` drives the slicing.
    #[test]
    fn driver_accumulator_advances_engine_deterministically() {
        let slices = 10u32;

        // Reference engine: step exactly N fixed slices with no input. Build it
        // through the same config path `level_setup` uses (default level +
        // settings + the default Marathon variant) so this test isolates the
        // *timestep* behavior, not config differences between the two engines.
        let config = engine_config_for_game(
            &LevelConfig::default(),
            &crate::settings::GameSettings::default(),
            crate::variant::ActiveVariant::default().0,
        );
        let mut reference = Engine::new(config, DEFAULT_SEED);
        for _ in 0..slices {
            reference.step(InputFrame {
                dt_seconds: SIM_DT_SECONDS,
                ..InputFrame::default()
            });
        }

        // Driven engine: same N slices, but through the real FixedUpdate schedule.
        // `FixedTimesteps(slices)` runs the fixed loop exactly that many times in
        // one `update()`, each with `Time::<Fixed>::delta() == SIM_DT_SECONDS`.
        let mut app = headless_app();
        tick_fixed(&mut app, slices);

        let driven = &app.world().resource::<LatestSnapshot>().0;
        assert_eq!(
            driven,
            &reference.snapshot(),
            "fixed-schedule driving matches direct fixed stepping"
        );
    }

    /// One press must produce exactly one engine action even when several fixed
    /// slices run in the same render frame. This replaces the removed
    /// `suppress_edges` system (the manual accumulator's per-slice edge guard):
    /// the property is now provided by latching the edge once in `PreUpdate` and
    /// draining + resetting [`PendingEdges`] on the first fixed slice. Held flags,
    /// by contrast, persist across slices so DAS/soft-drop keep advancing.
    #[test]
    fn single_press_yields_one_action_across_multiple_slices_in_a_frame() {
        let mut app = headless_app();
        // Spawn the first piece.
        tick_fixed(&mut app, 1);
        assert!(
            app.world().resource::<LatestSnapshot>().0.active.is_some(),
            "precondition: a piece is active before the hold press"
        );
        let holds_before = app.world().resource::<EngineState>().0.snapshot().hold;
        assert!(
            holds_before.is_none(),
            "precondition: hold slot starts empty"
        );

        // Press Hold (an edge action) and run THREE fixed slices in one frame.
        // The press is latched once in PreUpdate; only the first slice consumes it.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::ShiftLeft);
        tick_fixed(&mut app, 3);

        // A second hold in the same lifetime is illegal (the engine ignores it),
        // so a duplicated edge would be invisible in `hold`. Instead assert via
        // events: exactly one Held event was produced across the three slices.
        let held_events = app
            .world()
            .resource::<FrameEvents>()
            .0
            .iter()
            .filter(|e| matches!(e, EngineEvent::Held { .. }))
            .count();
        assert_eq!(
            held_events, 1,
            "one Hold press must yield exactly one Held action even across 3 slices"
        );
        assert!(
            app.world()
                .resource::<EngineState>()
                .0
                .snapshot()
                .hold
                .is_some(),
            "the held piece should now occupy the hold slot"
        );
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

    /// Drive a hard drop through the full app and verify the board reconciler
    /// mirrors the engine's locked board into StaticBlock entities. This exercises
    /// the snapshot -> board reconciliation path end to end.
    #[test]
    fn hard_drop_reconciles_locked_board_into_static_blocks() {
        let mut app = headless_app();

        // One fixed slice to spawn the first piece.
        tick_fixed(&mut app, 1);

        // Press Space (hard drop) and run another fixed slice; the latch carries
        // the edge from PreUpdate into the FixedUpdate step.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Space);
        tick_fixed(&mut app, 1);

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
