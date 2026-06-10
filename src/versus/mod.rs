//! Versus mode: two engines, attack routed between them.
//!
//! The design record is `docs/adr-versus-mode-ui.md`. The shape in one
//! paragraph: a match is two **seat entities** (engine + snapshot + events +
//! stats each), a [`Participant`] per seat saying who drives it (the local
//! keyboard or a [`ModelRegistry`](crate::ai::ModelRegistry) bot; a future
//! remote human is one more arm), and one fixed-update step that advances both
//! engines, routes every [`EngineEvent::AttackSent`] into the opposite seat's
//! pending queue, and ends the match when a seat dies. The engine already owns
//! every garbage *rule* (`docs/adr-versus-rules.md`); this module only routes.
//!
//! The match lives in [`GameState::Versus`] with its own
//! [`VersusPhase`] lifecycle (countdown → running ⇄ paused → over). The
//! single-player `level` module is untouched: nothing here reads or writes its
//! resources, and its systems never run in `Versus`.

use bevy::prelude::*;

use crate::engine::{
    Engine, EngineConfig, EngineEvent, EngineSnapshot, GoalSystem, LOCK_DOWN_SECONDS, MIN_LEVEL,
};
use crate::level::common::LevelConfig;
use crate::level::engine_bridge::{das_config_from_level, PendingEdges, SIM_DT_SECONDS};
use crate::player::{KeyboardController, PlayerController, RawKeyboardFrame};
use crate::GameState;

mod overlay;
mod render;

/// Lifecycle of a live match, as a sub-state of [`GameState::Versus`] — the
/// session (seat entities, boards, camera) is scoped to `Versus` itself, so
/// phase changes never despawn it. `Over` keeps the final boards on screen
/// under the result banner.
#[derive(SubStates, Clone, PartialEq, Eq, Hash, Debug, Default)]
#[source(GameState = GameState::Versus)]
pub enum VersusPhase {
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

/// Match configuration, written by the setup screen and read once when the
/// match spawns. `seed` is a test/replay override; a live match draws fresh
/// entropy per game (a rematch is a new deal, not a replay).
#[derive(Resource, Clone, Copy, Debug)]
pub struct VersusConfig {
    pub seats: [Participant; 2],
    pub seed: Option<u64>,
}

impl Default for VersusConfig {
    fn default() -> Self {
        Self {
            // You vs the Tier-2 beam — a mid-strength opener (registry index 1).
            seats: [Participant::Human, Participant::Bot { model: 1 }],
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
/// `PreUpdate`, like the single-player `FrameEvents`).
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

/// The bots seated this match, keyed by seat index. A non-send resource for
/// the same reason as the sandbox's `AiPlayer`: `AiController` is
/// `Send`-but-not-`Sync`, and the fixed-update driver runs on the main thread.
#[derive(Default)]
pub struct VersusBots(pub Vec<(usize, crate::ai::AiController)>);

/// How the match ended. Inserted exactly once, when a seat dies.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchOutcome {
    /// The winning seat, or `None` for a simultaneous draw.
    pub winner: Option<usize>,
}

/// Wall-clock match length (advances only while `Running`); the result banner
/// reports it.
#[derive(Resource, Default)]
pub struct MatchClock(pub f32);

/// The engine rules of a versus seat: flat level-1 gravity (no goal system —
/// pressure comes from the opponent, not the clock), the player's preview and
/// lock-down preferences applied symmetrically to both seats, and the standard
/// garbage cap.
fn versus_engine_config(settings: &crate::settings::GameSettings) -> EngineConfig {
    EngineConfig {
        board_width: 10,
        visible_height: 20,
        buffer_height: 20,
        preview_count: settings.next_count,
        lock_down_mode: settings.lock_down_mode,
        lock_down_seconds: LOCK_DOWN_SECONDS,
        starting_level: MIN_LEVEL,
        goal_system: GoalSystem::None,
        garbage_cap: EngineConfig::default().garbage_cap,
    }
}

pub struct VersusPlugin;

impl Plugin for VersusPlugin {
    fn build(&self, app: &mut App) {
        app.add_sub_state::<VersusPhase>()
            .init_resource::<VersusConfig>()
            .init_resource::<MatchClock>()
            // Self-sufficiency for headless tests (idempotent: `GamePlugin`
            // stays the canonical owner of the shared contracts).
            .init_resource::<crate::settings::GameSettings>()
            .init_resource::<crate::ai::ModelRegistry>()
            .init_resource::<LevelConfig>()
            .add_systems(OnEnter(GameState::Versus), versus_setup)
            .add_systems(OnExit(GameState::Versus), versus_teardown)
            .add_systems(
                PreUpdate,
                (clear_seat_events, latch_human_input)
                    .chain()
                    .after(bevy::input::InputSystems)
                    .run_if(in_state(VersusPhase::Running)),
            )
            .add_systems(
                FixedUpdate,
                versus_step.run_if(in_state(VersusPhase::Running)),
            )
            .add_systems(
                Update,
                (advance_match_clock, detect_match_end)
                    .chain()
                    .run_if(in_state(VersusPhase::Running)),
            )
            .add_plugins(render::VersusRenderPlugin)
            .add_plugins(overlay::VersusOverlayPlugin);
    }
}

/// Spawn the match: two seat entities (same engine seed — identical bags are
/// the guideline fairness convention; the hole streams stay decorrelated by
/// the engine's own salt) and one controller per bot seat. Exclusive because
/// bot controllers go into a non-send resource.
fn versus_setup(world: &mut World) {
    let config = *world.resource::<VersusConfig>();
    let settings = world.resource::<crate::settings::GameSettings>().clone();
    let engine_config = versus_engine_config(&settings);

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

    let mut bots = VersusBots::default();
    let das = das_config_from_level(world.resource::<LevelConfig>());

    for (index, participant) in config.seats.into_iter().enumerate() {
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
            DespawnOnExit(GameState::Versus),
        ));
        if let Some(human) = human {
            seat.insert(human);
        }
    }

    world.insert_non_send_resource(bots);
    world.insert_resource(MatchClock::default());
    world.remove_resource::<MatchOutcome>();
}

/// Drop the bots and the outcome when the session ends. Seat entities are
/// `DespawnOnExit(GameState::Versus)`-scoped, so Bevy tears those down.
fn versus_teardown(world: &mut World) {
    world.remove_non_send_resource::<VersusBots>();
    world.remove_resource::<MatchOutcome>();
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

fn versus_step(mut seats: SeatStepQuery, mut bots: NonSendMut<VersusBots>) {
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

/// End the match when a seat dies. Reads the published snapshots
/// (authoritative) rather than racing the event list; both dead in the same
/// slice is a draw.
fn detect_match_end(
    seats: Query<(&Seat, &SeatSnapshot)>,
    mut commands: Commands,
    mut next: ResMut<NextState<VersusPhase>>,
) {
    let mut dead = [false; 2];
    for (seat, snapshot) in &seats {
        if seat.index < 2 {
            dead[seat.index] = snapshot.0.game_over.is_some();
        }
    }
    if !dead[0] && !dead[1] {
        return;
    }
    let winner = match (dead[0], dead[1]) {
        (true, true) => None,
        (true, false) => Some(1),
        (false, true) => Some(0),
        (false, false) => unreachable!(),
    };
    info!("versus over: winner {winner:?}");
    commands.insert_resource(MatchOutcome { winner });
    next.set(VersusPhase::Over);
}

/// Restart the match in place (rematch): despawn the seat entities and rerun
/// the spawn path. Used by the result overlay; a fresh seed is drawn (a
/// rematch is a new deal).
// TODO(versus overlay): the result banner's Rematch action is the production
// caller; the allow comes off when it lands (this strike, flow pass).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn restart_match(world: &mut World) {
    let seats: Vec<Entity> = world
        .query_filtered::<Entity, With<Seat>>()
        .iter(world)
        .collect();
    for entity in seats {
        world.entity_mut(entity).despawn();
    }
    versus_teardown(world);
    versus_setup(world);
    // Render roots rebuild from the fresh seats on their next reconcile pass.
    world
        .resource_mut::<NextState<VersusPhase>>()
        .set(VersusPhase::Countdown);
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

    /// A headless versus app on a frozen clock: enter `Versus`, force the
    /// phase to `Running` (skipping the countdown), and advance only via
    /// explicit fixed slices.
    fn headless_versus_app(config: VersusConfig) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
            .init_state::<GameState>()
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(test_assets())
            .insert_resource(config)
            .add_plugins(VersusPlugin);
        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::Versus);
        app.update(); // queue the transition
        app.update(); // apply Versus + run setup
        app.world_mut()
            .resource_mut::<NextState<VersusPhase>>()
            .set(VersusPhase::Running);
        app.update(); // apply the phase
        app
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

    fn bot_match(seed: u64) -> VersusConfig {
        VersusConfig {
            // Greedy DT-20 on both seats: fast and deterministic.
            seats: [Participant::Bot { model: 0 }, Participant::Bot { model: 0 }],
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
    fn setup_seats_two_engines_with_the_same_deal() {
        let mut app = headless_versus_app(bot_match(7));
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
            let mut app = headless_versus_app(bot_match(seed));
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
        let mut app = headless_versus_app(bot_match(7));
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
            if app.world().get_resource::<MatchOutcome>().is_some() {
                break;
            }
        }
        assert!(
            routed,
            "a clear on seat 0 must queue garbage against seat 1"
        );
    }

    #[test]
    fn a_dead_seat_ends_the_match_with_the_survivor_winning() {
        let mut app = headless_versus_app(bot_match(7));
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
            if app.world().get_resource::<MatchOutcome>().is_some() {
                break;
            }
        }
        let outcome = *app
            .world()
            .get_resource::<MatchOutcome>()
            .expect("48 queued lines must kill seat 1 well within 10 seconds");
        assert_eq!(outcome.winner, Some(0), "the survivor takes the match");
        // The phase transition queued by `detect_match_end` applies on the
        // next state-transition pass — run one more frame before asserting.
        app.update();
        assert_eq!(
            app.world().resource::<State<VersusPhase>>().get(),
            &VersusPhase::Over,
            "the match parks in Over (boards stay on screen)"
        );
    }

    #[test]
    fn the_engine_queue_survives_the_bots_blinded_poll() {
        // The blinding is a strip on the snapshot handed to the bot; the
        // engine-side queue must remain intact (it still rises by rule, and
        // the UI's meter renders from the published snapshot).
        let mut app = headless_versus_app(bot_match(7));
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
        let mut app = headless_versus_app(bot_match(7));
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
            app.world().resource::<State<VersusPhase>>().get(),
            &VersusPhase::Countdown,
            "a rematch re-runs the countdown"
        );
    }

    #[test]
    fn leaving_versus_tears_down_seats_and_bots() {
        let mut app = headless_versus_app(bot_match(7));
        assert!(app.world().get_non_send_resource::<VersusBots>().is_some());

        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::MainMenu);
        app.update();
        app.update();

        assert!(
            app.world().get_non_send_resource::<VersusBots>().is_none(),
            "bots are dropped with the session"
        );
        let seats = app.world_mut().query::<&Seat>().iter(app.world()).count();
        assert_eq!(seats, 0, "seat entities are state-scoped");
    }
}
