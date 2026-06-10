//! `tetr_online` — a guideline Tetris built on Bevy.
//!
//! The crate is split along the ADR-7 boundary: [`engine`] is the
//! engine-agnostic rule core (no Bevy types), and everything else is the Bevy
//! host that drives it. The `level` module owns the in-game loop (stepping the
//! engine in `FixedUpdate` and reconciling snapshots into the ECS world);
//! `screens` and `features` provide menus and presentation; [`player`]
//! translates input into engine [`InputFrame`]s; [`storage`], [`settings`],
//! [`high_scores`], and [`variant`] handle persistence and run configuration.
//! The flat `tetr_online::` re-export surface below exposes the engine API the
//! tests and host build against.

use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;
#[cfg(feature = "dev")]
use bevy_egui::{egui, EguiContext, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext};
#[cfg(feature = "dev")]
use bevy_inspector_egui::{bevy_inspector, DefaultInspectorConfigPlugin};

// Engine-agnostic core (ADR-7), now the `tetr-core` crate: re-export `engine` and
// `player` so the host's existing `crate::engine::…` / `crate::player::…` paths keep
// resolving unchanged.
pub use tetr_core::{engine, player};

/// Game-side AI: the engine-agnostic `tetr-core::ai` plus the Bevy `sandbox`.
pub mod ai;
/// Play-evaluation harness (feature `arena`). Dev/research tooling — off by
/// default so it never compiles into the shipped game.
#[cfg(feature = "arena")]
pub mod arena;
mod assets;
pub(crate) mod features;
pub mod high_scores;
pub(crate) mod level;
pub(crate) mod postfx;
mod screens;
pub mod settings;
pub mod storage;
pub(crate) mod ui;
pub mod variant;
pub(crate) mod vfx;

pub use crate::engine::{
    apply_grounded_move_or_rotation, breaks_back_to_back, classify_t_spin, fall_speed_seconds,
    fixed_goal_for_level, goal_for_level, is_block_out, is_lock_out, is_top_out,
    qualifies_for_back_to_back, soft_drop_speed_seconds, t_spin_corners, variable_goal_for_level,
    variable_goal_units, ActivePiece, ActivePieceSnapshot, Engine, EngineConfig, EngineEvent,
    EngineScoreAction, EngineSnapshot, GameOverStatus, GoalProgress, GoalSystem, InputFrame,
    LockDownMode, PieceAction, PieceRotation, PieceType, RotationDirection, SnapshotCell,
    TSpinCorners, TSpinKind, EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS, MAX_LEVEL, MIN_LEVEL,
};
use crate::level::LevelPlugin;

/// Top-level screen the app is on. Drives which plugins' systems run and which
/// UI is spawned. Flow: `Loading` (asset load) -> `Title` -> `MainMenu`, with
/// `ModeSelect`/`Options`/`Help`/`HighScores` reachable from the menu, `Playing`
/// the active game, and `GameOver` after it. Pausing is a sub-state of `Playing`
/// (see [`PauseState`]), not a sibling state, so it never exits the session.
#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub enum GameState {
    /// Asset loading; advances to [`GameState::Title`] when assets are ready.
    #[default]
    Loading,
    /// Splash/title screen; any key advances to the main menu.
    Title,
    /// Root navigation menu (Play / Options / Help / High Scores).
    MainMenu,
    /// Choose which AI model drives the Watch-AI session (Watch-AI path only,
    /// before mode select). See [`screens`]/`model_select` + [`ai::ModelRegistry`].
    ModelSelect,
    /// Choose a [`variant::Variant`] (Marathon/Sprint/Ultra) before playing.
    ModeSelect,
    /// Settings screen (filled by the options feature).
    Options,
    /// Controls/about screen (filled by the help feature).
    Help,
    /// Leaderboards (filled by the high-scores feature).
    HighScores,
    /// Active gameplay (formerly `InGame`). The engine is authoritative here.
    /// Pause is the [`PauseState`] sub-state of this, so the session persists
    /// across pause/resume.
    Playing,
    /// Post-game results; offers restart / back to menu.
    GameOver,
}

/// Pause sub-state of [`GameState::Playing`].
///
/// Pause is modeled as a **sub-state of the active game**, not a sibling
/// `GameState`, so toggling it never exits `Playing`. The engine, board, camera,
/// and HUD are all scoped to `OnEnter(GameState::Playing)` /
/// `DespawnOnExit(GameState::Playing)`, so they survive a pause/resume round-trip
/// with no rebuild.
///
/// (A previous design used a `Playing | Paused` **computed** state for this scope,
/// but a `ComputedStates` re-runs its `OnEnter`/`OnExit` on *every* source change —
/// including identity transitions where the computed value is unchanged — so
/// pausing despawned and rebuilt the whole session, restarting the game on every
/// resume. A sub-state's transitions fire only when the sub-state itself changes.)
///
/// Per-frame gameplay systems (the engine driver, reconcilers, score/UI updates)
/// run on `in_state(PauseState::Running)`, so the simulation freezes while paused
/// while every gameplay entity stays alive.
#[derive(SubStates, Clone, PartialEq, Eq, Hash, Debug, Default)]
#[source(GameState = GameState::Playing)]
pub enum PauseState {
    /// The game is live; the simulation advances.
    #[default]
    Running,
    /// Paused: the overlay is shown and the simulation is frozen.
    Paused,
}

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameState>()
            .add_loading_state(
                LoadingState::new(GameState::Loading)
                    .load_collection::<crate::assets::GameAssets>()
                    .continue_to_state(GameState::Title),
            )
            // Shared M1 contracts (defined once, read everywhere).
            .init_resource::<crate::settings::GameSettings>()
            .init_resource::<crate::variant::ActiveVariant>()
            .init_resource::<crate::variant::VariantProgress>()
            .init_resource::<crate::high_scores::HighScores>()
            // The Watch-AI model registry (linear DT-20 + ported CC2 models on
            // greedy / beam / best-first). Read by the model-select screen and the
            // sandbox.
            .init_resource::<crate::ai::ModelRegistry>()
            // Runtime visual-FX toggles (all-on; the dev panel flips them live).
            .init_resource::<crate::vfx::VfxToggles>()
            .register_type::<crate::vfx::VfxToggles>()
            // Reflection registration for the shared M1 contracts (canonical
            // owner). Inner non-engine types embedded in these (Keybinds,
            // GameAction, Variant, HighScore) are registered so the inspector can
            // descend into them. Engine-typed fields (LockDownMode) are
            // `#[reflect(ignore)]`d at the field to preserve the engine boundary.
            .register_type::<crate::settings::GameSettings>()
            .register_type::<crate::settings::Keybinds>()
            .register_type::<crate::settings::GameAction>()
            .register_type::<crate::variant::ActiveVariant>()
            .register_type::<crate::variant::Variant>()
            .register_type::<crate::variant::VariantProgress>()
            .register_type::<crate::high_scores::HighScores>()
            .register_type::<crate::high_scores::HighScore>()
            .insert_resource(crate::storage::StorageResource(
                crate::storage::default_storage(),
            ))
            // Gameplay + screen-shell + feature plugins.
            .add_plugins(LevelPlugin)
            // AI sandbox (AI3.6): the "Watch AI" mode that drives a gameplay
            // session with the bot. Owns the `AiSandbox` flag the menus arm/clear
            // and the AI engine driver; the level driver reads the flag to choose
            // keyboard vs. AI, so adding this is purely additive.
            .add_plugins(crate::ai::AiSandboxPlugin)
            .add_plugins(crate::screens::ScreensPlugin)
            .add_plugins(crate::features::FeaturesPlugin)
            // Render-pipeline visual effects (CRT pass; bloom on capable builds).
            .add_plugins(crate::postfx::PostFxPlugin);

        #[cfg(debug_assertions)]
        {
            app.add_plugins(FrameTimeDiagnosticsPlugin::default())
                .add_plugins(LogDiagnosticsPlugin::default());
        }

        // Dev-only ECS inspector overlay (egui). Behind the `dev` cargo feature
        // (not `debug_assertions`) so release builds never compile egui — keeps
        // the size-optimized wasm clean. We drive it via the core manual API
        // (`DefaultInspectorConfigPlugin` + a `ui_for_world` window) rather than
        // `quick::WorldInspectorPlugin`: the `quick` module requires the
        // inspector's `bevy_render` feature, which assumes a 3D-capable Bevy this
        // curated 2D build doesn't enable (it fails to compile `generate_tangents`
        // and panics registering `GizmoConfigStore`). The window reads the
        // `register_type` registrations above to show entities / components /
        // resources by name. Run with `cargo run --features dev`.
        #[cfg(feature = "dev")]
        {
            app.add_plugins(EguiPlugin::default())
                .add_plugins(DefaultInspectorConfigPlugin)
                .add_systems(EguiPrimaryContextPass, dev_inspector_ui)
                // Live per-effect toggles for the visual-FX stack.
                .add_systems(EguiPrimaryContextPass, crate::vfx::vfx_debug_panel);
        }
    }
}

/// Draw the dev ECS inspector window — entities, components, resources, assets.
/// Only compiled with the `dev` feature; reads the `register_type` registry (see
/// `GamePlugin::build`) so custom components/resources show by name.
#[cfg(feature = "dev")]
fn dev_inspector_ui(world: &mut World) {
    // Clone the primary egui context handle so the world borrow is released
    // before we hand `world` to the inspector below.
    let Ok(mut egui_context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single(world)
        .cloned()
    else {
        return;
    };
    egui::Window::new("Inspector").show(egui_context.get_mut(), |ui| {
        egui::ScrollArea::both().show(ui, |ui| {
            bevy_inspector::ui_for_world(world, ui);
        });
    });
}
