//! The session: every game is N seat entities on one pipeline.
//!
//! The design record is `docs/adr-versus-mode-ui.md`. The shape in one
//! paragraph: a session is `SessionMode::seat_count` **seat entities**
//! (engine + snapshot + events + stats each), a `Participant` per seat
//! saying who drives it (the local keyboard or a
//! [`ModelRegistry`](crate::ai::ModelRegistry) bot; a future remote human is
//! one more arm), and one fixed-update step that advances every engine,
//! routes [`EngineEvent::AttackSent`] into the opposite seat's pending queue,
//! and ends the session when a seat dies or a solo goal is met. The engine
//! owns every garbage *rule* (`docs/adr-versus-rules.md`); this module only
//! routes. Single-player is the one-seat case: same step, same render, with
//! the variant's rules folded in through the engine-config seam.
//!
//! The session lives in [`GameState::Session`] with its own
//! `SessionPhase` lifecycle (countdown → running ⇄ paused → over).

use bevy::prelude::*;

use crate::GameState;
use crate::engine::{
    Engine, EngineConfig, EngineEvent, EngineSnapshot, GoalSystem, LOCK_DOWN_SECONDS, MIN_LEVEL,
};
use crate::level::common::LevelConfig;
use crate::level::engine_bridge::{PendingEdges, SIM_DT_SECONDS, das_config_from_level};
use crate::player::{KeyboardController, PlayerController, RawKeyboardFrame};

mod feel;
mod overlay;
pub(crate) mod render;

/// Lifecycle of a live session, as a sub-state of [`GameState::Session`] —
/// the session (seat entities, boards, camera) is scoped to the outer state,
/// so phase changes never despawn it. `Over` keeps the final boards on screen
/// under the result banner.
#[derive(SubStates, Clone, PartialEq, Eq, Hash, Debug, Default)]
#[source(GameState = GameState::Session)]
pub enum SessionPhase {
    /// 3-2-1-GO. Engines hold (no first spawn) until the countdown ends, so
    /// both first pieces appear simultaneously.
    #[default]
    Countdown,
    /// The match is live; both engines step every fixed slice.
    Running,
    /// Frozen, overlay shown. Pausing a local match is inherently mutual.
    Paused,
    /// A seat died; the result banner is up over the final boards.
    Over,
}

/// Who drives a seat. Deliberately an open set: a remote human is one more
/// arm producing an `InputFrame` per slice, not a redesign.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Participant {
    /// The local keyboard (at most one seat in v1 — there is one keyboard).
    Human,
    /// A bot from the [`ModelRegistry`](crate::ai::ModelRegistry) catalog.
    Bot {
        /// Index into the registry.
        model: usize,
    },
}

/// What a session is FOR — the per-mode rules the seat machinery reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionMode {
    /// A solo variant run on one seat (a human's game, or Watch-AI's bot):
    /// variant rules (leveling gravity, goals, end conditions), the score HUD,
    /// high-score capture for human runs.
    Solo { variant: crate::variant::Variant },
    /// The versus match on two seats: flat gravity, attack exchange, meters.
    Versus,
}

impl SessionMode {
    /// Seats this mode plays with (the prefix of [`SessionConfig::seats`]).
    pub fn seat_count(self) -> usize {
        match self {
            SessionMode::Solo { .. } => 1,
            SessionMode::Versus => 2,
        }
    }
}

/// Session configuration, written by the menus and read once when the session
/// spawns. `seats[..mode.seat_count()]` are the live seats. `seed` is a
/// test/replay override; a live session draws fresh entropy per game (a
/// rematch/retry is a new deal, not a replay).
#[derive(Resource, Clone, Copy, Debug)]
pub struct SessionConfig {
    pub seats: [Participant; 2],
    pub mode: SessionMode,
    pub seed: Option<u64>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            // You vs the Tier-2 beam — a mid-strength opener (registry index 1).
            seats: [Participant::Human, Participant::Bot { model: 1 }],
            mode: SessionMode::Versus,
            seed: None,
        }
    }
}

/// A seat at the match: `0` is the left board, `1` the right.
#[derive(Component, Clone, Copy, Debug)]
pub struct Seat {
    pub index: usize,
}

/// The seat's authoritative simulation.
#[derive(Component)]
pub struct SeatEngine(pub Engine);

/// The seat's snapshot, republished after every step (post-routing, so the
/// pending meter a frame renders already includes attack that arrived this
/// slice).
#[derive(Component)]
pub struct SeatSnapshot(pub EngineSnapshot);

/// Engine events of every slice that ran this render frame (cleared in
/// `PreUpdate`, refilled by the slices that run after it).
#[derive(Component, Default)]
pub struct SeatEvents(pub Vec<EngineEvent>);

/// Running match totals for the HUD and the result banner.
#[derive(Component, Default, Clone, Copy)]
pub struct SeatStats {
    /// Net attack lines actually sent (post-cancellation).
    pub attack_sent: u32,
    /// Garbage lines that rose onto this board.
    pub garbage_taken: u32,
}

/// The local keyboard, seated. Owns the same latch discipline as the
/// single-player driver (`PendingEdges` + a staged held frame, drained once
/// per slice) and the player-side DAS state machine — per seat, so a second
/// local keymap later is a second component, not a refactor.
#[derive(Component)]
pub struct HumanSeat {
    controller: KeyboardController,
    held: RawKeyboardFrame,
    edges: PendingEdges,
}

/// The bots seated this match, keyed by seat index. A non-send resource
/// because `AiController` is `Send`-but-not-`Sync`, and the fixed-update
/// driver runs on the main thread anyway.
#[derive(Default)]
pub struct SessionBots(pub Vec<(usize, crate::ai::AiController)>);

/// How the session ended. Inserted exactly once.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionOutcome {
    /// Versus: the winning seat, or `None` for a simultaneous draw.
    Versus { winner: Option<usize> },
    /// Solo: the run ended — `completed` iff the variant's goal was met
    /// (vs a top-out). The banner reads the final snapshot for the numbers.
    Solo { completed: bool },
}

/// Wall-clock session length (advances only while `Running`); the result
/// banner reports it, and solo variants read it for time-limit/ranking rules.
#[derive(Resource, Default)]
pub struct MatchClock(pub f32);

/// The rank a finished solo run earned on the leaderboard (None = did not
/// place, or a bot/unqualified run). Written when the session ends; the
/// result banner reads it.
#[derive(Resource, Clone, Copy)]
pub struct SoloRecorded(pub Option<usize>);

/// The engine rules for this session's seats, by mode. Versus: flat level-1
/// gravity (no goal system — pressure comes from the opponent, not the clock)
/// with the player's preview/lock-down preferences applied symmetrically.
/// Solo: the variant's own rules through the same config seam single-player
/// always used (leveling gravity, goal systems, variant overrides).
fn session_engine_config(
    mode: SessionMode,
    level: &LevelConfig,
    settings: &crate::settings::GameSettings,
) -> EngineConfig {
    match mode {
        SessionMode::Solo { variant } => {
            crate::level::engine_bridge::engine_config_for_game(level, settings, variant)
        }
        SessionMode::Versus => EngineConfig {
            board_width: 10,
            visible_height: 20,
            preview_count: settings.next_count,
            lock_down_mode: settings.lock_down_mode,
            lock_down_seconds: LOCK_DOWN_SECONDS,
            starting_level: MIN_LEVEL,
            goal_system: GoalSystem::None,
            garbage_cap: EngineConfig::default().garbage_cap,
        },
    }
}

pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        // The simulation contract: FixedUpdate runs at SIM_HZ, and every engine
        // step is stamped SIM_DT_SECONDS. Both sides of that equation live here
        // because this plugin owns the stepping; Bevy's default fixed clock is
        // 64 Hz, which would run a 1/60-stamped simulation ~7% fast.
        app.insert_resource(Time::<Fixed>::from_hz(
            crate::level::engine_bridge::SIM_HZ as f64,
        ))
        .add_sub_state::<SessionPhase>()
        .init_resource::<SessionConfig>()
        .init_resource::<MatchClock>()
        // Self-sufficiency for headless tests (idempotent: `GamePlugin`
        // stays the canonical owner of the shared contracts).
        .init_resource::<crate::settings::GameSettings>()
        .init_resource::<crate::ai::ModelRegistry>()
        .init_resource::<LevelConfig>()
        .init_resource::<crate::high_scores::HighScores>()
        .add_systems(OnEnter(GameState::Session), session_setup)
        .add_systems(OnExit(GameState::Session), session_teardown)
        .add_systems(
            PreUpdate,
            (clear_seat_events, latch_human_input)
                .chain()
                .after(bevy::input::InputSystems)
                .run_if(in_state(SessionPhase::Running)),
        )
        .add_systems(
            FixedUpdate,
            session_step.run_if(in_state(SessionPhase::Running)),
        )
        .add_systems(
            Update,
            (advance_match_clock, check_solo_end)
                .chain()
                .run_if(in_state(SessionPhase::Running)),
        )
        .add_systems(OnEnter(SessionPhase::Over), record_solo_run)
        // A press latched in the same render frame as a pause (a frame
        // that ran zero slices) must not fire on the first slice after a
        // resume, minutes later.
        .add_systems(OnEnter(SessionPhase::Running), reset_human_latch)
        .add_plugins(render::SessionRenderPlugin)
        .add_plugins(overlay::SessionOverlayPlugin)
        .add_plugins(feel::SessionFeelPlugin);
    }
}

/// Spawn the match: two seat entities (same engine seed — identical bags are
/// the guideline fairness convention; the hole streams stay decorrelated by
/// the engine's own salt) and one controller per bot seat. Exclusive because
/// bot controllers go into a non-send resource.
fn session_setup(world: &mut World) {
    let config = *world.resource::<SessionConfig>();
    let settings = world.resource::<crate::settings::GameSettings>().clone();
    let engine_config = {
        let level = world.resource::<LevelConfig>();
        session_engine_config(config.mode, level, &settings)
    };

    // Fresh deal per match: app-clock entropy unless a test/replay pinned it.
    // (Headless tests freeze the clock, so they pin the seed explicitly.)
    let seed = config.seed.unwrap_or_else(|| {
        world
            .resource::<Time<Real>>()
            .elapsed()
            .subsec_nanos()
            .wrapping_mul(0x9E37_79B9) as u64
            ^ world.resource::<Time<Real>>().elapsed().as_nanos() as u64
    });
    info!("versus match: seed {seed}, seats {:?}", config.seats);

    let mut bots = SessionBots::default();
    let das = das_config_from_level(world.resource::<LevelConfig>());

    for (index, participant) in config
        .seats
        .into_iter()
        .take(config.mode.seat_count())
        .enumerate()
    {
        // Resolve the participant's driver before spawning (bot construction
        // reads the registry resource, which can't overlap the spawn borrow).
        let mut human = None;
        match participant {
            Participant::Human => {
                human = Some(HumanSeat {
                    controller: KeyboardController::new(das),
                    held: RawKeyboardFrame::default(),
                    edges: PendingEdges::default(),
                });
            }
            Participant::Bot { model } => {
                let registry = world.resource::<crate::ai::ModelRegistry>();
                // An out-of-range model (a stale config) falls back to the
                // first catalog entry rather than panicking mid-spawn.
                let controller = registry.build(model).unwrap_or_else(|| {
                    warn!("versus: model {model} not in the registry; using entry 0");
                    registry.build(0).expect("the registry is never empty")
                });
                bots.0.push((index, controller));
            }
        }

        let engine = Engine::new(engine_config.clone(), seed);
        let snapshot = engine.snapshot();
        let mut seat = world.spawn((
            Seat { index },
            SeatEngine(engine),
            SeatSnapshot(snapshot),
            SeatEvents::default(),
            SeatStats::default(),
            DespawnOnExit(GameState::Session),
        ));
        if let Some(human) = human {
            seat.insert(human);
        }
    }

    world.insert_non_send_resource(bots);
    world.insert_resource(MatchClock::default());
    world.remove_resource::<SessionOutcome>();
    world.remove_resource::<SoloRecorded>();
}

/// Drop the bots and the outcome when the session ends. Seat entities are
/// `DespawnOnExit(GameState::Session)`-scoped, so Bevy tears those down.
fn session_teardown(world: &mut World) {
    world.remove_non_send_resource::<SessionBots>();
    world.remove_resource::<SessionOutcome>();
}

/// Clear every seat's per-frame event buffer before this frame's fixed slices
/// append to it (the versus mirror of the single-player `clear_frame_events`).
fn clear_seat_events(mut seats: Query<&mut SeatEvents>) {
    for mut events in &mut seats {
        events.0.clear();
    }
}

/// Sample the keyboard once per render frame for the human seat: latch edges,
/// stage held flags — the same drop/dup-safe discipline as the single-player
/// `latch_input`, but stored on the seat so a second keymap later is data, not
/// architecture.
fn latch_human_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    settings: Res<crate::settings::GameSettings>,
    mut humans: Query<&mut HumanSeat>,
) {
    for mut human in &mut humans {
        let raw = crate::features::options::keyboard_input_from_keybinds(
            &keyboard,
            &settings.keybinds,
            settings.hold_enabled,
            SIM_DT_SECONDS,
        );
        human.edges.latch(&raw);
        human.held = raw;
    }
}

/// Advance both engines one fixed slice and route the attack between them.
///
/// Order, per slice: **step both seats first** (collecting each seat's
/// events), **then** route every `AttackSent` into the opposite seat's queue,
/// **then** publish snapshots. Stepping before routing makes the exchange
/// symmetric — an attack always lands exactly one slice after the clear that
/// sent it, in both directions, regardless of seat order. Routing into a seat
/// that died this slice is inert by engine rule (dead engines accept no
/// garbage), and a dying lock sends nothing — the driver never second-guesses
/// events.
type SeatStepQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Seat,
        &'static mut SeatEngine,
        &'static mut SeatSnapshot,
        &'static mut SeatEvents,
        &'static mut SeatStats,
        Option<&'static mut HumanSeat>,
    ),
>;

fn session_step(
    mut seats: SeatStepQuery,
    bots: Option<NonSendMut<SessionBots>>,
    outcome: Option<Res<SessionOutcome>>,
    config: Res<SessionConfig>,
    mut commands: Commands,
    mut next: ResMut<NextState<SessionPhase>>,
) {
    // `Option`: every legal path into `Running` passes through the `OnEnter`
    // that seats the bots, but a dev-inspector state poke does not — be inert
    // rather than panic the app.
    let Some(mut bots) = bots else {
        return;
    };
    // The match ended in an earlier slice of this same render frame (the
    // `Over` transition only applies between frames): the outcome is settled,
    // so later slices must not keep playing into it.
    if outcome.is_some() {
        return;
    }
    // Phase 1: step every seat with its participant's frame.
    let mut slice_events: [Vec<EngineEvent>; 2] = [Vec::new(), Vec::new()];
    for (seat, mut engine, snapshot, _, _, human) in &mut seats {
        let events = match human {
            Some(mut human) => {
                let mut input = human.held;
                input.dt_seconds = SIM_DT_SECONDS;
                human.edges.drain_onto(&mut input);
                human.controller.set_input(input);
                let frame = human.controller.poll(&snapshot.0);
                human.edges.reset();
                engine.0.step(frame)
            }
            None => {
                let Some((_, bot)) = bots.0.iter_mut().find(|(i, _)| *i == seat.index) else {
                    continue; // a seat with no driver idles (should not happen)
                };
                // The bot plays BLIND to the pending queue — deliberately. The
                // experimental record (versus_climb header) shows the aware
                // search is decisively worse under pressure with today's
                // weights (the mispricing finding), and blindness also denies
                // a bot the perfect hole information a human can't see. The
                // engine still cancels and rises by rule regardless.
                let mut snap = engine.0.snapshot();
                snap.pending_garbage.clear();
                let frame = bot.poll(&snap);
                engine.0.step(frame)
            }
        };
        if seat.index < 2 {
            slice_events[seat.index] = events;
        }
    }

    // Phase 2: route attack across seats (0 → 1, 1 → 0), symmetrically.
    let attack: [u32; 2] = [sent_lines(&slice_events[0]), sent_lines(&slice_events[1])];
    for (seat, mut engine, _, _, mut stats, _) in &mut seats {
        let incoming = attack[1 - seat.index.min(1)];
        if incoming > 0 {
            engine.0.queue_garbage(incoming);
        }
        stats.attack_sent += attack[seat.index.min(1)];
        stats.garbage_taken += slice_events[seat.index]
            .iter()
            .map(|e| match e {
                EngineEvent::GarbageInserted { lines } => *lines,
                _ => 0,
            })
            .sum::<u32>();
    }

    // Phase 3: publish post-routing snapshots and the frame's events.
    for (seat, engine, mut snapshot, mut events, _, _) in &mut seats {
        snapshot.0 = engine.0.snapshot();
        events.0.extend(slice_events[seat.index].iter().cloned());
    }

    // Phase 4: death check, **per slice** — several slices can run in one
    // render frame (catch-up after a hitch), and the first death ends the
    // match in *its* slice. A frame-granular check would keep both engines
    // playing to the end of the frame and could score "both died this frame"
    // as a draw when one seat in fact outlived the other; a draw is only a
    // death in the *same slice*. The commands apply between slices, so the
    // guard above freezes everything after this one.
    let mut dead = [false; 2];
    for (seat, _, snapshot, _, _, _) in &seats {
        if seat.index < 2 {
            dead[seat.index] = snapshot.0.game_over.is_some();
        }
    }
    if dead[0] || dead[1] {
        let outcome = match config.mode {
            SessionMode::Versus => {
                let winner = match (dead[0], dead[1]) {
                    (true, true) => None,
                    (true, false) => Some(1),
                    _ => Some(0),
                };
                info!("versus over: winner {winner:?}");
                SessionOutcome::Versus { winner }
            }
            // Solo: a death is an incomplete run (the variant goal ends runs
            // via `check_solo_end`, not here).
            SessionMode::Solo { .. } => SessionOutcome::Solo { completed: false },
        };
        commands.insert_resource(outcome);
        next.set(SessionPhase::Over);
    }
}

/// Solo only: end the run when the active variant's goal/time condition is
/// met (Marathon level cap, Sprint line target, Ultra time limit). Death is
/// the step's job; this is the *successful* ending.
fn check_solo_end(
    config: Res<SessionConfig>,
    clock: Res<MatchClock>,
    seats: Query<&SeatSnapshot, With<Seat>>,
    outcome: Option<Res<SessionOutcome>>,
    mut commands: Commands,
    mut next: ResMut<NextState<SessionPhase>>,
) {
    let SessionMode::Solo { variant } = config.mode else {
        return;
    };
    if outcome.is_some() {
        return;
    }
    let Some(snapshot) = seats.iter().next() else {
        return;
    };
    let def = variant.def();
    if crate::variant::end_condition_met(&def, &snapshot.0, clock.0) {
        info!("solo run complete ({})", def.display_name);
        commands.insert_resource(SessionOutcome::Solo { completed: true });
        next.set(SessionPhase::Over);
    }
}

/// Solo only: when the run ends, file a HUMAN run for the leaderboard (bot
/// seats — Watch-AI — never rank against the player) and stash the rank for
/// the result banner.
fn record_solo_run(
    config: Res<SessionConfig>,
    clock: Res<MatchClock>,
    seats: Query<(&SeatSnapshot, Option<&HumanSeat>), With<Seat>>,
    storage: Option<Res<crate::storage::StorageResource>>,
    mut scores: ResMut<crate::high_scores::HighScores>,
    mut commands: Commands,
) {
    let SessionMode::Solo { variant } = config.mode else {
        return;
    };
    let Some((snapshot, human)) = seats.iter().next() else {
        return;
    };
    let rank = if human.is_some() {
        crate::features::high_scores::record(&snapshot.0, clock.0, variant, &mut scores, &storage)
    } else {
        None
    };
    commands.insert_resource(SoloRecorded(rank));
}

/// Total net attack in a slice's events.
fn sent_lines(events: &[EngineEvent]) -> u32 {
    events
        .iter()
        .map(|e| match e {
            EngineEvent::AttackSent { lines } => *lines,
            _ => 0,
        })
        .sum()
}

/// The match clock ticks while the match runs (shown on the result banner).
fn advance_match_clock(time: Res<Time>, mut clock: ResMut<MatchClock>) {
    clock.0 += time.delta_secs();
}

/// Drop any stale latched edges/held flags when the match (re)starts running —
/// a hard drop pressed in the instant of pausing stays latched through the
/// whole pause otherwise (the single-player latch has the same hazard; here it
/// is closed).
fn reset_human_latch(mut humans: Query<&mut HumanSeat>) {
    for mut human in &mut humans {
        human.edges.reset();
        human.held = RawKeyboardFrame::default();
    }
}

/// Restart the match in place (rematch): despawn the seat entities and rerun
/// the spawn path. Used by the result overlay; a fresh seed is drawn (a
/// rematch is a new deal).
pub(crate) fn restart_match(world: &mut World) {
    let seats: Vec<Entity> = world
        .query_filtered::<Entity, With<Seat>>()
        .iter(world)
        .collect();
    for entity in seats {
        world.entity_mut(entity).despawn();
    }
    session_teardown(world);
    session_setup(world);
    // Render roots rebuild from the fresh seats on their next reconcile pass.
    world
        .resource_mut::<NextState<SessionPhase>>()
        .set(SessionPhase::Countdown);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::GameAssets;
    use bevy::state::app::StatesPlugin;
    use bevy::time::TimeUpdateStrategy;
    use core::time::Duration;

    fn test_assets() -> GameAssets {
        GameAssets {
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
            font_body: default(),
        }
    }

    /// A headless versus app on a frozen clock: enter `Versus`, force the
    /// phase to `Running` (skipping the countdown), and advance only via
    /// explicit fixed slices.
    fn headless_session_app(config: SessionConfig) -> App {
        let mut app = bare_session_app(config);
        app.world_mut()
            .resource_mut::<NextState<SessionPhase>>()
            .set(SessionPhase::Running);
        app.update(); // apply the phase
        app
    }

    /// The harness without the phase override: enters `Versus` and leaves the
    /// match on its natural opening phase (the countdown).
    fn bare_session_app(config: SessionConfig) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
            .init_state::<GameState>()
            .insert_resource(ButtonInput::<KeyCode>::default())
            // `focus_navigation` (the pause/result menus) reads the mouse
            // accumulator that `bevy_input` provides in a real app.
            .insert_resource(bevy::input::mouse::AccumulatedMouseMotion::default())
            .insert_resource(test_assets())
            .insert_resource(config)
            .add_plugins(SessionPlugin);
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Session);
        app.update(); // queue the transition
        app.update(); // apply Versus + run setup
        app
    }

    /// The simulation contract has two halves: every engine step stamps
    /// `SIM_DT_SECONDS`, and `FixedUpdate` must run at `SIM_HZ` for those
    /// stamps to be wall-true. Bevy defaults the fixed clock to 64 Hz, so a
    /// session app that forgets to seed it simulates ~7% fast (a 500 ms lock
    /// delay elapses in ~469 ms). The fixed-timestep harness can't see the
    /// rate (it advances whole slices), hence this direct pin.
    #[test]
    fn the_session_app_steps_at_sim_hz() {
        let app = bare_session_app(solo_human(7));
        assert_eq!(
            app.world().resource::<Time<Fixed>>().timestep(),
            Duration::from_secs_f64(1.0 / f64::from(crate::level::engine_bridge::SIM_HZ)),
        );
    }

    /// Run exactly `n` fixed slices, in chunks of 10 per render frame:
    /// `Time<Virtual>::max_delta` (250 ms) silently clamps anything larger,
    /// so one `FixedTimesteps(600)` update would run only ~15 slices.
    fn tick_fixed(app: &mut App, n: u32) {
        let mut left = n;
        while left > 0 {
            let chunk = left.min(10);
            app.insert_resource(TimeUpdateStrategy::FixedTimesteps(chunk));
            app.update();
            left -= chunk;
        }
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    }

    /// The reviewer's race hypothesis, settled empirically: a NATURAL death
    /// (not a manual phase poke) must end a solo run as completed=false, with
    /// the goal checker unable to overwrite it in the same frame. Garbage
    /// overflow kills the seat mid-schedule exactly like a real top-out.
    #[test]
    fn a_natural_solo_death_records_incomplete() {
        let mut app = headless_session_app(solo_human(7));
        tick_fixed(&mut app, 1); // spawn
        // Bury the seat: queue far more garbage than the board holds, then a
        // clear-less lock rises it (the human seat plays neutral frames, so
        // gravity locks the piece eventually).
        {
            let mut seats = app.world_mut().query::<&mut SeatEngine>();
            let mut engine = seats.iter_mut(app.world_mut()).next().unwrap();
            engine.0.queue_garbage(48);
        }
        // Hard-drop every other frame (press/release so each edge latches):
        // every lock is clear-less, rising 8 queued lines, so the overflow
        // death arrives within a handful of pieces.
        for i in 0..240 {
            {
                let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
                if i % 2 == 0 {
                    keys.press(KeyCode::Space);
                } else {
                    keys.release(KeyCode::Space);
                }
            }
            tick_fixed(&mut app, 1);
            if app.world().get_resource::<SessionOutcome>().is_some() {
                break;
            }
        }
        let outcome = *app
            .world()
            .get_resource::<SessionOutcome>()
            .expect("48 queued lines must kill the solo seat");
        assert_eq!(
            outcome,
            SessionOutcome::Solo { completed: false },
            "a death is an incomplete run — and nothing may overwrite it"
        );
        // Drop the key AND its edges: this harness has no InputPlugin, so
        // nothing clears just_pressed — a stale Space edge would Select the
        // result banner's Retry row every frame. (Production clears edges in
        // PreUpdate; en route this stale edge DID drive death → banner →
        // retry end-to-end, which was a nice accidental flow check.)
        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.release(KeyCode::Space);
            keys.clear();
        }
        // Run several more frames: the settled outcome must stay settled.
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SessionOutcome>(),
            SessionOutcome::Solo { completed: false },
            "the outcome must not be rewritten after the run ends"
        );
    }

    /// The natural GOAL path: an Ultra time-limit expiry ends the run as
    /// completed=true (check_solo_end's verdict), exercising the clock-based
    /// ending with no death anywhere near.
    #[test]
    fn a_solo_goal_completion_records_complete() {
        let mut app = headless_session_app(SessionConfig {
            seats: [Participant::Human, Participant::Bot { model: 0 }],
            mode: SessionMode::Solo {
                variant: crate::variant::Variant::Ultra,
            },
            seed: Some(7),
        });
        // Ultra's limit is minutes of sim time; instead of ticking it out,
        // pre-load the clock just under the limit and tick across it.
        let limit = match crate::variant::Variant::Ultra.def().end_condition {
            crate::variant::EndCondition::TimeLimit(limit) => limit,
            _ => unreachable!("Ultra is time-limited"),
        };
        app.world_mut().resource_mut::<MatchClock>().0 = limit - 0.01;
        for _ in 0..30 {
            tick_fixed(&mut app, 1);
            if app.world().get_resource::<SessionOutcome>().is_some() {
                break;
            }
        }
        assert_eq!(
            *app.world().resource::<SessionOutcome>(),
            SessionOutcome::Solo { completed: true },
            "a met time limit is a completed run"
        );
    }

    /// Solo pause conceals the field (the anti-pause-think rule); resume
    /// restores it. Versus pause is covered by the overlay's own tests.
    #[test]
    fn solo_pause_conceals_the_board() {
        let mut app = headless_session_app(solo_human(7));
        tick_fixed(&mut app, 2);

        app.world_mut()
            .resource_mut::<NextState<SessionPhase>>()
            .set(SessionPhase::Paused);
        app.update();
        {
            let mut roots = app.world_mut().query::<(&render::BoardRoot, &Visibility)>();
            for (_, visibility) in roots.iter(app.world()) {
                assert_eq!(
                    *visibility,
                    Visibility::Hidden,
                    "a solo pause must conceal the field"
                );
            }
        }
        app.world_mut()
            .resource_mut::<NextState<SessionPhase>>()
            .set(SessionPhase::Running);
        app.update();
        let mut roots = app.world_mut().query::<(&render::BoardRoot, &Visibility)>();
        for (_, visibility) in roots.iter(app.world()) {
            assert_eq!(
                *visibility,
                Visibility::Inherited,
                "resume must reveal the field"
            );
        }
    }

    fn solo_human(seed: u64) -> SessionConfig {
        SessionConfig {
            seats: [Participant::Human, Participant::Bot { model: 0 }],
            mode: SessionMode::Solo {
                variant: crate::variant::Variant::Marathon,
            },
            seed: Some(seed),
        }
    }

    /// THE schedule-determinism pin: N fixed slices through the real schedule
    /// must leave a keys-untouched human seat's engine byte-identical to a
    /// directly-stepped reference (neutral frames, same dt) — gravity and
    /// lock-down advance independent of render framing.
    #[test]
    fn solo_schedule_matches_direct_engine_stepping() {
        let slices = 10u32;
        let mut app = headless_session_app(solo_human(7));
        tick_fixed(&mut app, slices);

        let level = LevelConfig::default();
        let settings = crate::settings::GameSettings::default();
        let config = session_engine_config(
            SessionMode::Solo {
                variant: crate::variant::Variant::Marathon,
            },
            &level,
            &settings,
        );
        let mut reference = Engine::new(config, 7);
        for _ in 0..slices {
            reference.step(crate::engine::InputFrame {
                dt_seconds: SIM_DT_SECONDS,
                ..Default::default()
            });
        }

        let mut seats = app.world_mut().query::<&SeatSnapshot>();
        let snapshot = seats.iter(app.world()).next().expect("one seat");
        assert_eq!(
            snapshot.0,
            reference.snapshot(),
            "schedule-driven solo play must match direct fixed stepping"
        );
    }

    /// One press = one action even when several fixed slices run in a single
    /// render frame — the edge-latch property.
    #[test]
    fn one_press_yields_one_action_across_slices() {
        let mut app = headless_session_app(solo_human(7));
        tick_fixed(&mut app, 1); // spawn the first piece

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::ShiftLeft); // the default Hold bind
        tick_fixed(&mut app, 3); // three slices, one render frame each chunk

        let mut seats = app.world_mut().query::<&SeatEngine>();
        let engine = seats.iter(app.world()).next().expect("one seat");
        assert!(
            engine.0.snapshot().hold.is_some(),
            "the held piece should occupy the hold slot"
        );
        // A second hold in the same lifetime is illegal; a duplicated edge
        // would be invisible in `hold`. The latch discipline is pinned by the
        // versus-side reset tests; here the observable is one successful hold.
    }

    /// Pause must freeze WITHOUT despawning: the seat entities and the
    /// engine's exact state survive a pause/resume round-trip.
    #[test]
    fn pause_preserves_the_session() {
        let mut app = headless_session_app(solo_human(7));
        tick_fixed(&mut app, 30);

        let before: Vec<_> = {
            let mut q = app.world_mut().query::<(&Seat, &SeatSnapshot)>();
            q.iter(app.world())
                .map(|(s, snap)| (s.index, snap.0.clone()))
                .collect()
        };

        app.world_mut()
            .resource_mut::<NextState<SessionPhase>>()
            .set(SessionPhase::Paused);
        app.update();
        tick_fixed(&mut app, 10); // time passes; nothing may step
        app.world_mut()
            .resource_mut::<NextState<SessionPhase>>()
            .set(SessionPhase::Running);
        app.update();

        let after: Vec<_> = {
            let mut q = app.world_mut().query::<(&Seat, &SeatSnapshot)>();
            q.iter(app.world())
                .map(|(s, snap)| (s.index, snap.0.clone()))
                .collect()
        };
        assert_eq!(before, after, "pause must freeze, not rebuild or advance");
    }

    /// Watch-AI: a one-seat BOT session plays the game with no keyboard
    /// input at all.
    #[test]
    fn a_solo_bot_seat_drives_the_engine() {
        let mut app = headless_session_app(SessionConfig {
            seats: [Participant::Bot { model: 0 }, Participant::Bot { model: 0 }],
            mode: SessionMode::Solo {
                variant: crate::variant::Variant::Marathon,
            },
            seed: Some(7),
        });
        let mut locked = false;
        for _ in 0..240 {
            tick_fixed(&mut app, 1);
            let mut seats = app.world_mut().query::<&SeatSnapshot>();
            if !seats
                .iter(app.world())
                .next()
                .unwrap()
                .0
                .board_cells
                .is_empty()
            {
                locked = true;
                break;
            }
        }
        assert!(locked, "the bot must lock a piece with no keyboard input");
    }

    /// The leaderboard rules: a finished HUMAN solo run files (and stashes
    /// its rank); a BOT solo run never does.
    #[test]
    fn solo_recording_files_humans_and_skips_bots() {
        for (config, expect_recorded) in [
            (solo_human(7), true),
            (
                SessionConfig {
                    seats: [Participant::Bot { model: 0 }, Participant::Bot { model: 0 }],
                    mode: SessionMode::Solo {
                        variant: crate::variant::Variant::Marathon,
                    },
                    seed: Some(7),
                },
                false,
            ),
        ] {
            let mut app = headless_session_app(config);
            tick_fixed(&mut app, 5);
            app.world_mut()
                .resource_mut::<NextState<SessionPhase>>()
                .set(SessionPhase::Over);
            app.update();

            let scores = app.world().resource::<crate::high_scores::HighScores>();
            let table = scores.table(crate::variant::Variant::Marathon);
            assert_eq!(
                !table.is_empty(),
                expect_recorded,
                "human files, bot does not (expect_recorded={expect_recorded})"
            );
        }
    }

    fn bot_match(seed: u64) -> SessionConfig {
        SessionConfig {
            // Greedy DT-20 on both seats: fast and deterministic.
            seats: [Participant::Bot { model: 0 }, Participant::Bot { model: 0 }],
            mode: SessionMode::Versus,
            seed: Some(seed),
        }
    }

    fn snapshots(app: &mut App) -> Vec<(usize, EngineSnapshot)> {
        let mut all: Vec<(usize, EngineSnapshot)> = app
            .world_mut()
            .query::<(&Seat, &SeatSnapshot)>()
            .iter(app.world())
            .map(|(seat, snap)| (seat.index, snap.0.clone()))
            .collect();
        all.sort_by_key(|(i, _)| *i);
        all
    }

    #[test]
    fn the_countdown_hands_off_to_running() {
        // Enter Versus WITHOUT forcing the phase: the match must open on the
        // countdown, hold the engines (no first spawn), and hand off to
        // Running on its own once the beats elapse.
        let mut app = bare_session_app(bot_match(7));
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Countdown,
            "a match opens on the countdown"
        );

        // 200 ms per frame (under `Time<Virtual>::max_delta`'s 250 ms clamp):
        // the 2.6 s of beats elapse within 16 frames.
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
            200,
        )));
        for _ in 0..16 {
            app.update();
            let snaps = snapshots(&mut app);
            if app.world().resource::<State<SessionPhase>>().get() == &SessionPhase::Countdown {
                assert!(
                    snaps[0].1.active.is_none(),
                    "engines hold during the countdown (no first spawn)"
                );
            }
        }
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Running,
            "the countdown hands off to Running by itself"
        );
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
        tick_fixed(&mut app, 1);
        assert!(
            snapshots(&mut app)[0].1.active.is_some(),
            "the first piece spawns on the first Running slice"
        );
    }

    #[test]
    fn pause_freezes_the_match_and_esc_resumes() {
        let mut app = headless_session_app(bot_match(7));
        tick_fixed(&mut app, 60);

        // The pause keybind (Escape by default) freezes the whole match.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Escape);
        app.update(); // pause_on_keybind queues Paused
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear_just_pressed(KeyCode::Escape);
        app.update(); // the transition applies
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Paused
        );

        let frozen = snapshots(&mut app);
        tick_fixed(&mut app, 30);
        assert_eq!(
            snapshots(&mut app),
            frozen,
            "a paused match must not advance either engine"
        );

        // Esc again resumes (the pause menu's Back action).
        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.release(KeyCode::Escape);
            keys.clear_just_released(KeyCode::Escape);
            keys.press(KeyCode::Escape);
        }
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear_just_pressed(KeyCode::Escape);
        app.update();
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Running,
            "Esc on the pause menu resumes the match"
        );
    }

    #[test]
    fn setup_seats_two_engines_with_the_same_deal() {
        let mut app = headless_session_app(bot_match(7));
        tick_fixed(&mut app, 1);

        let snaps = snapshots(&mut app);
        assert_eq!(snaps.len(), 2, "a match seats exactly two engines");
        assert_eq!(
            snaps[0].1.next_queue, snaps[1].1.next_queue,
            "identical bags: the match measures placement, not draw luck"
        );
        assert!(
            snaps[0].1.active.is_some(),
            "the first slice spawns the first piece"
        );
    }

    #[test]
    fn bots_drive_both_seats_and_the_match_is_deterministic() {
        let run = |seed: u64, slices: u32| {
            let mut app = headless_session_app(bot_match(seed));
            tick_fixed(&mut app, slices);
            snapshots(&mut app)
        };
        let a = run(11, 600);
        let b = run(11, 600);
        assert_eq!(a, b, "same seed, same match, byte for byte");
        // Progress check via the lines counter (board cells fluctuate near
        // zero in a mirror greedy match — it digs singles constantly).
        assert!(
            a[0].1.lines > 0 && a[1].1.lines > 0,
            "after 10 seconds both bots have cleared lines"
        );
        let c = run(12, 600);
        assert_ne!(a, c, "a different seed deals a different match");
    }

    #[test]
    fn attack_routes_to_the_opposite_seat() {
        // The engine-level attack accounting is pinned in tetr-core; what this
        // test owns is the driver's cross-wiring. Singles send nothing (the
        // research record: greedy mirror duels never produce net attack), so
        // hand seat 0 a ready-made Tetris well — four rows filled except the
        // right column. Greedy finds the multi-line clear within a couple of
        // pieces (verified at this seed), the engine emits `AttackSent`, and
        // the routed lines must appear against seat 1 (pending, or already
        // risen as garbage cells).
        let mut app = headless_session_app(bot_match(7));
        {
            let mut query = app.world_mut().query::<(&Seat, &mut SeatEngine)>();
            for (seat, mut engine) in query.iter_mut(app.world_mut()) {
                if seat.index == 0 {
                    for y in 0..4 {
                        for x in 0..9 {
                            engine.0.set_cell(
                                x,
                                y,
                                crate::engine::CellKind::Some(crate::engine::PieceType::J),
                            );
                        }
                    }
                }
            }
        }
        // Run until seat 1 has pending or risen garbage (the routed attack).
        let mut routed = false;
        for _ in 0..600 {
            tick_fixed(&mut app, 1);
            let snaps = snapshots(&mut app);
            let seat1 = &snaps[1].1;
            let garbage_cells = seat1.board_cells.iter().filter(|c| c.garbage).count();
            if seat1.pending_garbage_total() > 0 || garbage_cells > 0 {
                routed = true;
                break;
            }
            if app.world().get_resource::<SessionOutcome>().is_some() {
                break;
            }
        }
        assert!(
            routed,
            "a clear on seat 0 must queue garbage against seat 1"
        );
        // The feel pass: net attack leaving a board spawns a "+n" pop.
        let pops = app
            .world_mut()
            .query::<&feel::AttackPop>()
            .iter(app.world())
            .count();
        assert!(pops > 0, "sent attack shows a +n pop by the sender's board");
    }

    #[test]
    fn a_dead_seat_ends_the_match_with_the_survivor_winning() {
        let mut app = headless_session_app(bot_match(7));
        // Bury seat 1: queue far more garbage than the board holds; its next
        // clear-less lock rises it into a block-out.
        {
            let mut query = app.world_mut().query::<(&Seat, &mut SeatEngine)>();
            for (seat, mut engine) in query.iter_mut(app.world_mut()) {
                if seat.index == 1 {
                    for _ in 0..6 {
                        engine.0.queue_garbage(8);
                    }
                }
            }
        }
        for _ in 0..600 {
            tick_fixed(&mut app, 1);
            if app.world().get_resource::<SessionOutcome>().is_some() {
                break;
            }
        }
        let outcome = *app
            .world()
            .get_resource::<SessionOutcome>()
            .expect("48 queued lines must kill seat 1 well within 10 seconds");
        assert_eq!(
            outcome,
            SessionOutcome::Versus { winner: Some(0) },
            "the survivor takes the match"
        );
        // The phase transition queued by `detect_match_end` applies on the
        // next state-transition pass — run one more frame before asserting.
        app.update();
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Over,
            "the match parks in Over (boards stay on screen)"
        );
    }

    #[test]
    fn the_engine_queue_survives_the_bots_blinded_poll() {
        // The blinding is a strip on the snapshot handed to the bot; the
        // engine-side queue must remain intact (it still rises by rule, and
        // the UI's meter renders from the published snapshot).
        let mut app = headless_session_app(bot_match(7));
        {
            let mut query = app.world_mut().query::<(&Seat, &mut SeatEngine)>();
            for (seat, mut engine) in query.iter_mut(app.world_mut()) {
                if seat.index == 1 {
                    engine.0.queue_garbage(3);
                }
            }
        }
        tick_fixed(&mut app, 1);
        let snaps = snapshots(&mut app);
        // The published snapshot (what the UI's meter reads) still shows the
        // queue — blindness is the bot's, not the renderer's. (It may have
        // already risen if the first lock happened, hence the disjunction.)
        let seat1 = &snaps[1].1;
        let garbage_cells = seat1.board_cells.iter().filter(|c| c.garbage).count();
        assert!(
            seat1.pending_garbage_total() == 3 || garbage_cells > 0,
            "the engine-side queue survives the bot's blinded poll"
        );
    }

    #[test]
    fn restart_match_reseats_fresh_engines() {
        let mut app = headless_session_app(bot_match(7));
        tick_fixed(&mut app, 240); // four seconds in: boards have content
        let before = snapshots(&mut app);
        assert!(before[0].1.board_cells.len() + before[1].1.board_cells.len() > 0);

        restart_match(app.world_mut());
        app.update();

        let after = snapshots(&mut app);
        assert_eq!(after.len(), 2, "a rematch seats two engines again");
        assert!(
            after[0].1.board_cells.is_empty() && after[1].1.board_cells.is_empty(),
            "fresh engines: empty boards"
        );
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Countdown,
            "a rematch re-runs the countdown"
        );
    }

    #[test]
    fn the_renderer_mirrors_both_seats() {
        use render::{BoardRoot, LayerSeat, SeatMeter, VsLayer};

        let mut app = headless_session_app(bot_match(7));
        // Queue garbage against seat 1 so the meter has something to show.
        {
            let mut query = app.world_mut().query::<(&Seat, &mut SeatEngine)>();
            for (seat, mut engine) in query.iter_mut(app.world_mut()) {
                if seat.index == 1 {
                    engine.0.queue_garbage(3);
                    engine.0.queue_garbage(2);
                }
            }
        }
        tick_fixed(&mut app, 2); // spawn pieces; run the reconcilers

        let roots = app
            .world_mut()
            .query::<&BoardRoot>()
            .iter(app.world())
            .count();
        assert_eq!(roots, 2, "one board root per seat");

        // Each seat's falling layer carries the 4 cells of its active piece.
        let mut layers = app.world_mut().query::<(&VsLayer, &LayerSeat, &Children)>();
        for seat in 0..2 {
            let falling_cells = layers
                .iter(app.world())
                .find(|(l, s, _)| **l == VsLayer::Falling && s.0 == seat)
                .map(|(_, _, children)| children.len())
                .expect("each seat has a falling layer");
            assert_eq!(falling_cells, 4, "seat {seat}'s active piece renders");
        }

        // Seat 1's meter shows its two pending batches as two segments.
        let mut meters = app.world_mut().query::<(&SeatMeter, &Children)>();
        let segments = meters
            .iter(app.world())
            .find(|(m, _)| m.seat == 1)
            .map(|(_, children)| children.len())
            .expect("seat 1 has a meter");
        assert_eq!(segments, 2, "two queued batches render as two segments");
    }

    #[test]
    fn death_is_scored_in_its_slice_not_at_frame_end() {
        // Several fixed slices run per render frame after a hitch; the first
        // death must end the match in ITS slice. Bury both seats — they die a
        // few slices apart (seat 1 carries extra board cells, so its bot
        // plays a different, slower-dying game) — and the verdict must name a
        // winner: under frame-granular detection both would read dead at the
        // end of the chunk and mis-score a draw.
        let mut app = headless_session_app(bot_match(7));
        {
            let mut query = app.world_mut().query::<(&Seat, &mut SeatEngine)>();
            for (seat, mut engine) in query.iter_mut(app.world_mut()) {
                for _ in 0..6 {
                    engine.0.queue_garbage(8);
                }
                if seat.index == 1 {
                    // A small asymmetry so the deaths land in different slices.
                    engine.0.set_cell(
                        0,
                        0,
                        crate::engine::CellKind::Some(crate::engine::PieceType::J),
                    );
                }
            }
        }
        for _ in 0..60 {
            tick_fixed(&mut app, 10); // multi-slice frames, like a hitch
            if app.world().get_resource::<SessionOutcome>().is_some() {
                break;
            }
        }
        let outcome = *app
            .world()
            .get_resource::<SessionOutcome>()
            .expect("both seats buried: someone must die");
        assert!(
            matches!(outcome, SessionOutcome::Versus { winner: Some(_) }),
            "deaths slices apart in one frame must crown the survivor, not draw"
        );
    }

    #[test]
    fn the_rematch_request_rebuilds_the_match() {
        // The exact path the result banner's Rematch button takes: insert the
        // request resource; the exclusive applier reseats the match.
        let mut app = headless_session_app(bot_match(7));
        tick_fixed(&mut app, 240);
        assert!(
            snapshots(&mut app)[0].1.lines > 0 || !snapshots(&mut app)[0].1.board_cells.is_empty()
        );

        app.world_mut().insert_resource(overlay::RematchRequested);
        app.update(); // the applier reseats and queues Countdown
        app.update(); // the phase transition applies

        let after = snapshots(&mut app);
        assert!(
            after[0].1.board_cells.is_empty() && after[0].1.lines == 0,
            "a rematch deals a fresh match"
        );
        assert_eq!(
            app.world().resource::<State<SessionPhase>>().get(),
            &SessionPhase::Countdown
        );
    }

    #[test]
    fn leaving_versus_tears_down_seats_and_bots() {
        let mut app = headless_session_app(bot_match(7));
        assert!(app.world().get_non_send_resource::<SessionBots>().is_some());

        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::MainMenu);
        app.update();
        app.update();

        assert!(
            app.world().get_non_send_resource::<SessionBots>().is_none(),
            "bots are dropped with the session"
        );
        let seats = app.world_mut().query::<&Seat>().iter(app.world()).count();
        assert_eq!(seats, 0, "seat entities are state-scoped");
    }
}
