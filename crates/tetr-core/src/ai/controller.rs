//! The AI player controller: the model-agnostic shell.
//!
//! [`AiController`] is a [`PlayerController`]: it reads an [`EngineSnapshot`] and
//! returns the next [`InputFrame`], exactly like the keyboard controller, so it
//! drops into the same engine-driver seam the game's session seats use.
//!
//! It is the *shell* around an AI brain, and it is deliberately **model-agnostic**:
//! it knows nothing about search, evaluators, or weights — only a
//! [`Policy`](crate::ai::Policy), driven through a [`DecisionRunner`]. The same
//! shell drives a greedy search, a future beam, or a neural policy unchanged. It
//! owns the two concerns a brain has no opinion on:
//!
//! - **When** to act — a [`Handicap::reaction`] delay before playing a new piece.
//! - **How** to act — rendering the policy's chosen placement to engine input, one
//!   pulse per poll.
//!
//! The *other* half of the handicap — how *perfectly* to play — lives in the policy,
//! not here (a search softmax-samples a near-best, a net raises its temperature).
//! Keeping it there is what lets this shell stay model-blind.
//!
//! # One pulse per poll
//!
//! The engine consumes at most one action per `step` (a hold, a rotation, one
//! lateral cell, a soft/hard drop). So the controller never returns a compound
//! frame: it renders the chosen placement to a `Vec<InputFrame>` (via
//! [`placement_to_inputs`]) once per piece and **drains it one frame per poll**,
//! emitting a neutral frame (`dt` only) when the queue is empty or it is still
//! "reacting". This matches how the keyboard controller feeds the engine.
//!
//! # The poll state machine
//!
//! Each [`poll`](AiController::poll):
//! 1. **Detect a new piece.** If the active piece's identity changed (or the board
//!    did under the same piece), the current plan is stale: cancel the in-flight
//!    decision, clear the queued frames, and start a fresh think. The one
//!    *expected* change is exempt: a plan-initiated hold swap is pre-targeted
//!    when the plan is applied (see [`PieceSignature`]), so the bot's own hold
//!    never discards its own maneuver.
//! 2. **Pump.** Poll the runner every frame and buffer a finished [`Decision`].
//!    A cooperative venue ([`SlicedRunner`]) does its per-frame quantum of search
//!    *inside* that poll, so the thinking overlaps the reaction window below —
//!    this pump is what hides a heavy search's latency in the delay a human-like
//!    bot pays anyway.
//! 3. **React.** Accumulate the poll's `dt` into a reaction timer; while it is
//!    below [`Handicap::reaction`], emit neutral frames. The buffered decision is
//!    not *applied* until the delay elapses.
//! 4. **Apply.** Once the reaction elapses and a decision is buffered, render it
//!    to frames and enqueue them.
//! 5. **Emit.** Pop the next queued frame, or a neutral frame if none.
//!
//! # Determinism
//!
//! The shell adds no randomness and no clock — it only integrates the poll `dt`
//! (deterministic given the poll cadence). All AI randomness lives in the policy's
//! own seeded RNG. So a fixed `(engine seed, ai seed, handicap)` reproduces an
//! identical game every run — the determinism the headless benchmarks rely on.
//!
//! # No Bevy here
//!
//! Like the rest of the AI core this module has **no Bevy imports**: it is a plain
//! `PlayerController`, unit-testable by stepping an [`Engine`](crate::engine::Engine)
//! against it in a loop. Only the integration layer (`src/level`) is Bevy-aware.

use core::time::Duration;
use std::collections::VecDeque;

use crate::ai::eval::{Cc2Evaluator, Cc2Weights};
use crate::ai::handicap::Handicap;
use crate::ai::plan::placement_to_inputs;
use crate::ai::policy::{Decision, Policy, SearchPolicy};
use crate::ai::runner::{DecisionRunner, SlicedRunner, SyncRunner};
use crate::ai::search::{BestFirstPlanner, SearchBudget};
use crate::ai::state::SearchState;
use crate::engine::{EngineSnapshot, InputFrame, PieceType};
use crate::player::PlayerController;

/// A default seed for the AI's own RNG, distinct from the engine's default seed so
/// the two streams never accidentally align.
pub const DEFAULT_AI_SEED: u64 = 0xA1_5E_ED;

/// Total best-first node expansions per decision for [`AiController::attack`] —
/// the interactive quality dial. Headless benches run far higher (quality scales
/// with budget); this is the **window-capacity** point: the largest one-budget
/// value whose sliced think still completes inside the default 200 ms reaction
/// window on every platform, so the uplift from the sync-era 150 costs zero
/// pace. The arithmetic (pinned by `attack_budget_fits_the_reaction_window`):
/// the window is 12 polls at 60 Hz and the wasm worst-case quantum is 16
/// nodes/poll ⇒ 12 × 16 = 192. Beyond this, wasm pace degrades (the bot keeps
/// thinking past its reaction) — a conscious future operating-point decision,
/// not a constant bump.
const ATTACK_NODE_BUDGET: u32 = 192;

/// Ply cap for [`AiController::attack`]; best-first is depth-capped by the
/// visible queue, not width.
const ATTACK_DEPTH: u8 = 6;

/// An AI [`PlayerController`]: a model-agnostic shell that drives a
/// [`Policy`](crate::ai::Policy) (via a [`DecisionRunner`]) and feeds its chosen
/// placement to the engine one pulse per poll, with a reaction-delay handicap.
pub struct AiController {
    /// Where the decision is computed (synchronous by default; an off-thread runner
    /// drops in for a heavier policy with no other change).
    runner: Box<dyn DecisionRunner>,
    /// Frames left to emit for the current placement, front = next to play.
    plan: VecDeque<InputFrame>,
    /// Reaction delay before acting on a new piece (the shell-level handicap).
    reaction: Duration,
    /// Identity of the piece the current decision is for, to detect a new piece.
    /// `None` before the first piece is seen.
    planning_for: Option<PieceSignature>,
    /// Reaction time accumulated for the current piece (integrates poll `dt`).
    think_elapsed: f32,
    /// Whether a decision has been submitted for the current piece (so we submit
    /// once per piece, not every poll while reacting).
    submitted: bool,
    /// A decision the runner finished while the reaction window was still
    /// running, held until the controller may act on it.
    ready: Option<Decision>,
}

/// A cheap fingerprint of "which piece, on which board" so the controller can tell
/// a genuinely new planning situation from re-polling the same one.
///
/// Piece *type* alone is ambiguous (the next piece can repeat across a bag
/// boundary); pairing it with the locked-cell count distinguishes "same piece,
/// same board" (keep the current decision) from "new piece spawned" or "board
/// changed under us" (re-decide). Cheap to compute from a snapshot and `Eq`.
///
/// One transition is *expected* rather than new: a plan that begins with a hold
/// swaps the active piece one frame later. [`AiController::apply_decision`]
/// retargets `piece_type` to the post-hold piece when it enqueues such a plan, so
/// the swap the controller itself caused never discards the rest of the maneuver
/// (without the retarget, the bot re-pays the reaction delay *and* a full search
/// on every held piece).
#[derive(Clone, Copy, PartialEq, Eq)]
struct PieceSignature {
    piece_type: PieceType,
    /// Number of locked cells on the board (changes on any lock/clear).
    locked_cells: usize,
    /// Total pending-garbage lines queued against this player. An attack
    /// arriving **mid-think** (live versus: the opponent clears while we are
    /// still deciding) changes the planning situation as surely as a board
    /// change — the search modelled a queue that no longer exists — so it
    /// re-plans, paying the reaction delay again like a human noticing the
    /// meter jump. Cancellation can only happen on our own lock (which already
    /// changes `locked_cells`), so the total is a faithful change detector.
    /// Single-player and blinded bots always see `0` here: behaviour is
    /// byte-identical outside live versus.
    ///
    /// Known sharp edge for a future *aware* live bot (none ships yet): an
    /// arrival mid-maneuver also cancels the remaining frames, and if the
    /// piece is already grounded the re-paid reaction can race the lock-down
    /// timer into a timeout lock at a transit pose. If that bites, defer the
    /// pending-triggered replan until the plan queue is empty.
    pending_lines: u32,
}

impl PieceSignature {
    /// Derive the signature of a snapshot's current planning situation, if it has
    /// an active piece.
    fn of(snapshot: &EngineSnapshot) -> Option<Self> {
        let piece_type = snapshot.active.as_ref()?.piece_type;
        Some(Self {
            piece_type,
            locked_cells: snapshot.board_cells.len(),
            pending_lines: snapshot.pending_garbage_total(),
        })
    }
}

impl AiController {
    /// The shipped Tier-1 bot: a greedy [`SearchPolicy`] (default evaluator) behind
    /// a synchronous runner, with the given `handicap` and RNG `seed`.
    pub fn new(handicap: Handicap, seed: u64) -> Self {
        let policy = SearchPolicy::greedy(handicap.imperfection, seed);
        Self::with_policy(Box::new(policy), handicap.reaction)
    }

    /// A controller with the **default** handicap (a beatable opponent) and the
    /// default AI seed — the convenient construction for the game's bot seats.
    pub fn beatable() -> Self {
        Self::new(Handicap::default(), DEFAULT_AI_SEED)
    }

    /// The strongest shipped bot: a best-first graph search (per-root transposition)
    /// over the Cold Clear 2 evaluator with the APP-climbed attack weights
    /// ([`Cc2Weights::attack_tuned`]), behind the **cooperative** venue
    /// ([`SlicedRunner`]) so the per-piece search spreads across frames instead of
    /// stalling one — the same total work, no hitch, native and wasm alike. This
    /// is the **one home** for "the best model": the Watch-AI registry's
    /// best-first entry and the wasm embed both build through here, so the
    /// operating point can never fork between surfaces. The handicap dials
    /// (reaction delay + imperfection) still apply, so even the strongest brain
    /// stays beatable on demand.
    pub fn attack(handicap: Handicap, seed: u64) -> Self {
        let policy = SearchPolicy::new(
            Box::new(BestFirstPlanner::new()),
            Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
            SearchBudget::best_first(ATTACK_NODE_BUDGET, ATTACK_DEPTH),
            handicap.imperfection,
            seed,
        );
        Self::with_runner(
            Box::new(SlicedRunner::new(Box::new(policy))),
            handicap.reaction,
        )
    }

    /// The interactive-catalog construction: a (mind, evaluator, budget) triple
    /// under the **default** (beatable) handicap and AI seed, in the cooperative
    /// venue. One home for the menu-bot convention — beside [`attack`](Self::attack),
    /// the strongest model's same-venue home — so no game surface can fork the
    /// operating conventions from the core's.
    pub fn interactive(
        mind: Box<dyn crate::ai::Mind>,
        eval: Box<dyn crate::ai::Evaluator>,
        budget: SearchBudget,
    ) -> Self {
        let handicap = Handicap::default();
        let policy = SearchPolicy::new(mind, eval, budget, handicap.imperfection, DEFAULT_AI_SEED);
        Self::with_runner(
            Box::new(SlicedRunner::new(Box::new(policy))),
            handicap.reaction,
        )
    }

    /// A controller around an explicit [`Policy`], wrapped in the **blocking**
    /// runner ([`SyncRunner`]): the whole decision computes inline at submit.
    /// This is the headless construction — benchmarks and research bots, where
    /// no frame exists to hitch and exact full-budget decisions per poll are the
    /// contract. Interactive surfaces wrap their policy in a [`SlicedRunner`]
    /// via [`with_runner`](Self::with_runner) instead (as [`attack`](Self::attack)
    /// and the game's model registry do). `reaction` is the shell-level handicap;
    /// the policy carries its own imperfection + RNG.
    pub fn with_policy(policy: Box<dyn Policy>, reaction: Duration) -> Self {
        Self::with_runner(Box::new(SyncRunner::new(policy)), reaction)
    }

    /// A controller around an explicit [`DecisionRunner`] — the venue seam.
    /// Interactive surfaces pass a cooperative [`SlicedRunner`]; headless drivers
    /// usually reach for [`with_policy`](Self::with_policy) (blocking) instead.
    pub fn with_runner(runner: Box<dyn DecisionRunner>, reaction: Duration) -> Self {
        Self {
            runner,
            plan: VecDeque::new(),
            reaction,
            planning_for: None,
            think_elapsed: 0.0,
            submitted: false,
            ready: None,
        }
    }

    /// Begin planning for a freshly seen piece: cancel the in-flight decision, reset
    /// the reaction timer and queued frames, and remember what we are deciding for.
    fn begin_new_piece(&mut self, signature: PieceSignature) {
        self.runner.cancel();
        self.plan.clear();
        self.think_elapsed = 0.0;
        self.submitted = false;
        self.ready = None;
        self.planning_for = Some(signature);
    }

    /// Apply a finished decision: render the chosen placement to frames and
    /// enqueue them.
    fn apply_decision(&mut self, decision: Decision, obs: &SearchState) {
        match decision {
            // No legal placement: nothing to do (the engine tops out on its own);
            // leave the plan empty so we emit neutral frames.
            Decision::None => {}
            Decision::Place(placement) => {
                // A plan that begins with a hold swaps the active piece one frame
                // from now. That swap is the controller's own doing, not a new
                // planning situation: retarget the expected signature to the
                // post-hold piece (`placement.piece_type()` — the resting pose is
                // the swapped-in piece) so the maneuver continues uninterrupted.
                // The board is untouched by a hold, so `locked_cells` stands; if
                // the engine somehow doesn't swap, the mismatch re-plans — the
                // safe fallback either way.
                if placement.used_hold {
                    if let Some(signature) = self.planning_for.as_mut() {
                        signature.piece_type = placement.piece_type();
                    }
                }
                // Render against the board the maneuver happens on, from the active
                // piece's current pose — the inputs `placement_to_inputs` round-trips.
                let frames = placement_to_inputs(&obs.board.to_array2d(), &obs.active, &placement);
                self.plan = frames.into();
            }
        }
    }
}

impl PlayerController for AiController {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
        // (1) No active piece (pre-spawn / game over): nothing to decide; idle.
        let Some(signature) = PieceSignature::of(snapshot) else {
            return neutral();
        };

        // (1b) New planning situation? Reset and start a fresh think.
        if self.planning_for != Some(signature) {
            self.begin_new_piece(signature);
        }

        // If we still have queued frames for the current piece, keep draining them
        // (don't re-decide mid-maneuver). One pulse per poll. Maneuver frames carry
        // their own `dt == 0` so positioning advances no gravity — emit as-is.
        if let Some(frame) = self.plan.pop_front() {
            return frame;
        }

        // Build the observation once; reused for submit + render.
        let Some(obs) = SearchState::from_snapshot(snapshot) else {
            return neutral();
        };

        // (2) Submit once per piece, then pump the runner *every* poll — a
        // cooperative venue does its per-frame quantum inside `runner.poll()`, so
        // pumping during the reaction window below is precisely what overlaps the
        // thinking with the delay. A finished decision is buffered in `ready`
        // until the reaction allows acting on it.
        if !self.submitted {
            self.runner.submit(obs.clone());
            self.submitted = true;
        }
        if self.ready.is_none() {
            self.ready = self.runner.poll();
        }

        // (3) Accumulate the reaction timer; while it runs, emit neutral frames.
        self.think_elapsed += NOMINAL_DT;
        if self.think_elapsed < self.reaction.as_secs_f32() {
            return neutral(); // still "reacting"
        }

        // (4) Reaction elapsed: apply the buffered decision and emit its first
        // maneuver frame.
        if let Some(decision) = self.ready.take() {
            self.apply_decision(decision, &obs);
            if let Some(frame) = self.plan.pop_front() {
                return frame;
            }
        }

        // (5) Nothing to emit yet (decision not ready, or no legal move): idle.
        neutral()
    }
}

/// Nominal per-poll `dt` for the controller's reaction-timer integration: the fixed
/// sim slice at `SIM_HZ` (60 Hz). The driver steps the engine at that rate, so
/// integrating the reaction delay in these units paces it in real seconds. Maneuver
/// frames carry their *own* `dt == 0` (gravity-free positioning) and are emitted
/// unchanged; only neutral "reacting"/idle frames advance time.
const NOMINAL_DT: f32 = 1.0 / 60.0;

/// A neutral frame: advance one sim slice of time, press nothing. Emitted while
/// reacting or idle so gravity and the lock timer keep ticking between maneuvers.
fn neutral() -> InputFrame {
    InputFrame {
        dt_seconds: NOMINAL_DT,
        ..InputFrame::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    /// Step an engine with the controller for up to `max_frames`, returning how
    /// many pieces locked. Stops early on game over.
    ///
    /// Steps the controller's frame **as emitted** (no `dt` override) via the same
    /// contract as [`drive_engine`](crate::player::drive_engine): maneuver frames
    /// carry `dt == 0` so gravity does not desync the placement, neutral frames
    /// carry the sim slice.
    fn play(controller: &mut AiController, seed: u64, max_frames: usize) -> (usize, usize, bool) {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        let mut locks = 0usize;
        let mut topped_out = false;
        for _ in 0..max_frames {
            let snapshot = engine.snapshot();
            if snapshot.game_over.is_some() {
                topped_out = true;
                break;
            }
            // Count locks exactly via the engine's Locked events (robust to line
            // clears, which shrink the board-cell count).
            let events = engine.step(controller.poll(&snapshot));
            locks += events
                .iter()
                .filter(|e| matches!(e, crate::engine::EngineEvent::Locked { .. }))
                .count();
        }
        (locks, engine.snapshot().lines, topped_out)
    }

    #[test]
    fn controller_survives_and_clears_lines_with_default_weights() {
        // Play-QUALITY regression guard (not just "places pieces"): with the shipped
        // survival reward profile, the no-handicap bot keeps a clean board and clears
        // lines for a long run instead of stacking into a fast top-out. The old
        // Cold-Clear *downstacking* default — meant for a multi-ply beam — buried a
        // 1-ply greedy by ~piece 40-126, which this test would now catch.
        let mut controller = AiController::new(Handicap::perfect(), DEFAULT_AI_SEED);
        let (locks, lines, topped_out) = play(&mut controller, 7, 8_000);
        assert!(
            !topped_out,
            "survival weights must not top out within 8k frames (placed {locks}, cleared {lines})"
        );
        assert!(
            lines >= 40,
            "the bot should be clearing lines; cleared only {lines} (placed {locks})"
        );
    }

    #[test]
    fn controller_executes_the_planners_first_placement_faithfully() {
        // The controller's fidelity guarantee: it lands its first piece
        // exactly where the *planner* intends. We plan independently from the first
        // snapshot, then drive a perfect (instant, no-error) controller until the
        // piece locks, and assert the resulting board equals the planner's
        // simulated placement. (Plan-to-input pose fidelity is separately pinned by
        // `plan.rs`'s round-trip test; this proves the controller *sequences* it.)
        use crate::ai::eval::LinearEvaluator;
        use crate::ai::search::{think_to_completion, BestFirstPlanner, SearchBudget};
        use crate::ai::SearchState;
        use crate::engine::{lock_and_clear, Board, CellKind};

        let mut controller = AiController::new(Handicap::perfect(), DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();
        let config = snapshot.config.clone();

        // Planner's intended board after placing the first piece (hold-aware, like
        // the controller's own policy).
        let state = SearchState::from_snapshot(&snapshot).unwrap();
        let plan = think_to_completion(
            &mut BestFirstPlanner::new(),
            &state,
            &LinearEvaluator::default(),
            SearchBudget::single_ply(),
        )
        .expect("the first spawn has a legal placement");
        let mut intended = state.board.to_array2d();
        lock_and_clear(&plan.placement.piece, &mut intended);

        // Drive the controller until the board first changes (the piece locked).
        let mut guard = 0;
        loop {
            engine.step(controller.poll(&engine.snapshot()));
            let after = engine.snapshot();
            if !after.board_cells.is_empty() {
                let mut real = Board::with_top_margin(
                    config.board_width,
                    config.visible_height,
                    crate::engine::BUFFER_HEIGHT,
                );
                for c in &after.board_cells {
                    real.set(c.x, c.y, CellKind::Some(c.piece_type));
                }
                assert_eq!(
                    real.cell_coords(),
                    intended.cell_coords(),
                    "controller-executed placement diverged from the planner's intent"
                );
                break;
            }
            guard += 1;
            assert!(guard < 300, "the first piece never locked");
        }
    }

    #[test]
    fn neutral_frame_while_reacting() {
        // With a long reaction delay, the very first polls emit neutral frames (no
        // button pressed) — the bot is "reacting" before it acts.
        let handicap = Handicap {
            reaction: core::time::Duration::from_millis(500),
            ..Handicap::perfect()
        };
        let mut controller = AiController::new(handicap, DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 1);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();

        let frame = controller.poll(&snapshot);
        assert!(
            !pressed_anything(&frame),
            "first poll under a long reaction delay must be neutral"
        );
        assert!(frame.dt_seconds > 0.0, "neutral frames still advance time");
    }

    #[test]
    fn garbage_arriving_mid_think_restarts_the_reaction() {
        // Live versus: an attack landing in the queue while the bot is still
        // deciding is a new planning situation (the searched future modelled an
        // empty queue). The controller must drop the in-flight think and re-plan,
        // re-paying its reaction delay — like a human noticing the meter jump.
        let handicap = Handicap {
            reaction: core::time::Duration::from_millis(500), // 30 polls at 60 Hz
            ..Handicap::perfect()
        };
        let polls_until_first_action = |interrupt_at: Option<usize>| {
            let mut controller = AiController::new(handicap, DEFAULT_AI_SEED);
            let mut engine = Engine::new(EngineConfig::default(), 1);
            engine.step(InputFrame::default()); // spawn the first piece
            for poll in 0..120 {
                if interrupt_at == Some(poll) {
                    engine.queue_garbage(2); // the mid-think arrival
                }
                let frame = controller.poll(&engine.snapshot());
                if pressed_anything(&frame) {
                    return poll;
                }
                engine.step(frame);
            }
            panic!("controller never acted within 120 polls");
        };

        let undisturbed = polls_until_first_action(None);
        let interrupted = polls_until_first_action(Some(10));
        assert!(
            interrupted >= undisturbed + 8,
            "an arrival at poll 10 must restart the reaction \
             (undisturbed acted at {undisturbed}, interrupted at {interrupted})"
        );
    }

    #[test]
    fn determinism_same_seed_same_game() {
        // Two controllers with the same AI seed, driving two engines with the same
        // engine seed, must produce byte-identical games — the determinism the
        // headless benchmark relies on. Uses the default (imperfect) handicap so the
        // RNG path is actually exercised.
        let play_to_snapshot = |ai_seed: u64| {
            let mut controller = AiController::new(Handicap::default(), ai_seed);
            let mut engine = Engine::new(EngineConfig::default(), 42);
            for _ in 0..2_000 {
                let snap = engine.snapshot();
                if snap.game_over.is_some() {
                    break;
                }
                engine.step(controller.poll(&snap));
            }
            engine.snapshot()
        };

        let a = play_to_snapshot(DEFAULT_AI_SEED);
        let b = play_to_snapshot(DEFAULT_AI_SEED);
        assert_eq!(a, b, "same (engine, ai) seed must reproduce the game");
    }

    #[test]
    fn imperfection_uses_only_the_ai_rng() {
        // A controller with a high imperfection must still be fully reproducible from
        // its seed (RNG is owned + seeded, never the engine's). Different AI seeds
        // may diverge; the SAME seed must not.
        let high_error = Handicap {
            imperfection: 0.9,
            ..Handicap::perfect()
        };
        let run = |seed: u64| {
            let mut c = AiController::new(high_error, seed);
            let mut engine = Engine::new(EngineConfig::default(), 7);
            for _ in 0..1_500 {
                let snap = engine.snapshot();
                if snap.game_over.is_some() {
                    break;
                }
                engine.step(c.poll(&snap));
            }
            engine.snapshot().board_cells.len()
        };
        assert_eq!(
            run(123),
            run(123),
            "same AI seed reproduces even with errors"
        );
    }

    fn pressed_anything(frame: &InputFrame) -> bool {
        frame.left
            || frame.right
            || frame.soft_drop
            || frame.hard_drop
            || frame.rotate_clockwise
            || frame.rotate_counterclockwise
            || frame.hold
    }

    /// A policy that always plays a hold-using placement when one exists — the
    /// probe for the hold-replan fix (a plan beginning with `Move::Hold` must
    /// execute to the end, not be discarded when the swap lands).
    struct AlwaysHold;
    impl Policy for AlwaysHold {
        fn decide(&mut self, obs: &crate::ai::policy::Observation) -> Decision {
            crate::ai::search::hold_placements(obs)
                .into_iter()
                .find(|p| p.used_hold)
                .map(Decision::Place)
                .unwrap_or(Decision::None)
        }
    }

    #[test]
    fn a_hold_plan_executes_without_a_second_reaction() {
        // The double-reaction hazard: emitting the plan's leading hold frame
        // swaps the active piece, which an unguarded staleness check reads as a
        // "new piece" — discarding the queued maneuver and re-paying the reaction
        // delay plus a full search on every held piece. The pin: the swap is
        // pre-targeted, so the maneuver continues on the very next poll.
        let handicap = Handicap {
            reaction: core::time::Duration::from_millis(200),
            ..Handicap::perfect()
        };
        let mut controller = AiController::with_policy(Box::new(AlwaysHold), handicap.reaction);
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn the first piece

        let mut frames = Vec::new();
        let mut locked_at = None;
        for i in 0..120usize {
            let frame = controller.poll(&engine.snapshot());
            frames.push(frame.clone());
            let events = engine.step(frame);
            if locked_at.is_none()
                && events
                    .iter()
                    .any(|e| matches!(e, crate::engine::EngineEvent::Locked { .. }))
            {
                locked_at = Some(i);
            }
        }

        let hold_at = frames
            .iter()
            .position(|f| f.hold)
            .expect("the AlwaysHold bot must hold");
        let locked_at = locked_at.expect("the held placement must lock");
        assert!(locked_at > hold_at, "the lock comes from the held plan");
        assert!(
            pressed_anything(&frames[hold_at + 1]),
            "the maneuver must continue on the poll right after the hold swap \
             (a re-reaction gap means the plan was discarded)"
        );
        // One reaction window for the whole held placement: between the swap and
        // the lock there is never another reaction-length idle stretch.
        let neutral_gap = frames[hold_at + 1..=locked_at]
            .iter()
            .filter(|f| !pressed_anything(f))
            .count();
        assert!(
            neutral_gap < 12,
            "a second reaction window ({neutral_gap} idle frames) was paid after the hold"
        );
    }

    /// An attack-shaped policy at the **shipped** operating point (the same
    /// constants `AiController::attack` uses), so the venue gates below always
    /// test reality: raising the budget past the reaction window would fail
    /// them, forcing a conscious pace decision instead of a silent one.
    fn attack_policy(seed: u64) -> Box<dyn Policy> {
        Box::new(SearchPolicy::new(
            Box::new(BestFirstPlanner::new()),
            Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
            SearchBudget::best_first(ATTACK_NODE_BUDGET, ATTACK_DEPTH),
            Handicap::default().imperfection,
            seed,
        ))
    }

    #[test]
    fn attack_budget_fits_the_reaction_window() {
        // The pace-neutrality invariant behind ATTACK_NODE_BUDGET: the sliced
        // think must complete within the default reaction window on the slowest
        // platform (the wasm quantum), or the strongest bot silently slows down
        // on the web while looking fine natively.
        let window_polls = (Handicap::default().reaction.as_secs_f32() / NOMINAL_DT).round() as u32;
        assert!(
            ATTACK_NODE_BUDGET <= window_polls * crate::ai::runner::sliced::WASM_QUANTUM,
            "ATTACK_NODE_BUDGET ({ATTACK_NODE_BUDGET}) exceeds the wasm window \
             capacity ({window_polls} polls x {} nodes)",
            crate::ai::runner::sliced::WASM_QUANTUM,
        );
    }

    #[test]
    fn sliced_and_blocking_venues_play_the_identical_game() {
        // THE venue-swap gate: at the shipped operating point the sliced think
        // completes inside the default reaction window (the window-capacity
        // sizing pinned by attack_budget_fits_the_reaction_window), so swapping
        // the blocking venue for the cooperative one changes *where* the work
        // runs — never the game. Byte-identical engine snapshots over dozens of
        // pieces (clears, chain state, and several imperfection draws included;
        // 600 frames keeps the debug-build runtime reasonable — divergence would
        // surface at the first differing decision anyway).
        let reaction = Handicap::default().reaction;
        let play = |mut controller: AiController| {
            let mut engine = Engine::new(EngineConfig::default(), 42);
            for _ in 0..600 {
                let snap = engine.snapshot();
                if snap.game_over.is_some() {
                    break;
                }
                engine.step(controller.poll(&snap));
            }
            engine.snapshot()
        };

        let blocking = play(AiController::with_policy(attack_policy(7), reaction));
        let sliced = play(AiController::with_runner(
            Box::new(crate::ai::runner::SlicedRunner::with_quantum(
                attack_policy(7),
                16,
            )),
            reaction,
        ));
        assert_eq!(
            blocking, sliced,
            "the cooperative venue must reproduce the blocking venue's game"
        );
    }

    #[test]
    fn the_sliced_think_hides_inside_the_reaction_window() {
        // The latency-hiding mechanism: with the cooperative venue the controller
        // pumps a quantum per poll *during* the reaction delay, so the bot's
        // first input lands on the same poll as with the blocking venue — the
        // search costs no extra wall-clock, it just stops stalling a frame.
        let reaction = Handicap::default().reaction;
        let first_press_poll = |mut controller: AiController| {
            let mut engine = Engine::new(EngineConfig::default(), 7);
            engine.step(InputFrame::default()); // spawn the first piece
            for poll in 1..200 {
                let frame = controller.poll(&engine.snapshot());
                if pressed_anything(&frame) {
                    return poll;
                }
                engine.step(frame);
            }
            panic!("the bot never acted");
        };

        let blocking = first_press_poll(AiController::with_policy(attack_policy(7), reaction));
        let sliced = first_press_poll(AiController::with_runner(
            Box::new(crate::ai::runner::SlicedRunner::with_quantum(
                attack_policy(7),
                16,
            )),
            reaction,
        ));
        assert_eq!(
            blocking, sliced,
            "slicing must not delay the bot's first input past the reaction window"
        );
    }
}
