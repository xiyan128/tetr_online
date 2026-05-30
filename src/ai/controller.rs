//! The AI player controller (AI3.5).
//!
//! [`AiController`] is a [`PlayerController`]: it reads an [`EngineSnapshot`] and
//! returns the next [`InputFrame`], exactly like the keyboard controller, so it
//! drops into the same engine-driver seam (see `src/level/mod.rs`). It is the
//! layer that turns the pure search core (state → evaluator → movegen → planner →
//! plan-to-input) into a *played* game, adding the two things a search alone has
//! no opinion on: **when** to act (think-time) and **how perfectly** to act
//! (error injection).
//!
//! # One pulse per poll
//!
//! The engine consumes at most one action per `step` (a hold, a rotation, one
//! lateral cell, a soft/hard drop — see `engine/api.rs`). So the controller never
//! returns a compound frame: it computes a *plan* (a `Vec<InputFrame>` from
//! [`placement_to_inputs`]) once per piece and then **drains it one frame per
//! poll**, emitting a neutral frame (`dt` only) when the plan is empty or it is
//! still "thinking". This matches how the keyboard controller feeds the engine.
//!
//! # The poll state machine
//!
//! Each [`poll`](AiController::poll):
//! 1. **Detect a new piece.** If the active piece's identity changed (or the
//!    board did under the same piece — a topple), the current plan is stale: cancel
//!    the search, clear the queued frames, and start a fresh think.
//! 2. **Think.** Accumulate the poll's `dt` into a reaction timer; while it is
//!    below [`DifficultyConfig::think_time`], emit neutral frames. The search is
//!    *submitted* at the start of the think (so it can run off-thread during the
//!    delay) but its result is not *applied* until the delay elapses.
//! 3. **Apply.** Once think-time elapses and the runner has a plan, pick a
//!    placement (with error injection), render it to frames, and enqueue them.
//! 4. **Emit.** Pop the next queued frame, or a neutral frame if none.
//!
//! # Think-time, error injection, and the AI's own RNG
//!
//! All randomness uses [`self.rng`](AiController) — a seeded [`StdRng`] the
//! controller owns and seeds from a fixed `u64` (the same family the engine
//! generator uses, per the determinism contract). It **never** touches the
//! engine's RNG. Two places consume it:
//!
//! - **Error injection** ([`DifficultyConfig::error_rate`]). With probability
//!   `error_rate` the controller does not play the top placement; instead it
//!   softmax-samples among the top-N candidates (a near-best alternative, so a
//!   "mistake" looks plausible). At `error_rate == 0` it is deterministic.
//! - Think-time itself is *deterministic* given the poll cadence (it just
//!   integrates `dt`); only the placement choice is randomized. (Reaction-delay
//!   *jitter* is a future knob; the seam is here.)
//!
//! Because the RNG is seeded and the search is pure, a fixed
//! `(engine seed, ai seed, difficulty)` reproduces an identical game every run —
//! the determinism the M2 plan's headless benchmark relies on.
//!
//! # No Bevy here
//!
//! Like the rest of the AI core this module has **no Bevy imports**: it is a plain
//! `PlayerController`, unit-testable by stepping an [`Engine`](crate::engine::Engine)
//! against it in a loop. Only the integration layer (`src/level`) is Bevy-aware.

use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use std::collections::VecDeque;

use crate::ai::difficulty::DifficultyConfig;
use crate::ai::eval::{Evaluator, LinearEvaluator};
use crate::ai::plan::placement_to_inputs;
use crate::ai::runner::{ComputeRunner, SyncRunner};
use crate::ai::search::{GreedyPlanner, PlacementPlan, SearchBudget};
use crate::ai::state::SearchState;
use crate::engine::{EngineSnapshot, InputFrame, PieceType};
use crate::player::PlayerController;

/// A default seed for the controller's own RNG, distinct from the engine's
/// default seed so the two streams never accidentally align.
pub const DEFAULT_AI_SEED: u64 = 0xA1_5E_ED;

/// An AI [`PlayerController`]: searches for a placement and feeds it to the engine
/// one pulse per poll, with tunable think-time and error injection.
pub struct AiController {
    /// Where the search runs (synchronous by default; an off-thread runner can be
    /// swapped in for a Tier-2 beam with no other change).
    runner: Box<dyn ComputeRunner>,
    /// Frames left to emit for the current placement, front = next to play.
    plan: VecDeque<InputFrame>,
    /// The AI's own seeded RNG — error injection / tie-breaking only. Never the
    /// engine's.
    rng: StdRng,
    /// Tunable difficulty (think-time, error rate, depth, node budget).
    difficulty: DifficultyConfig,
    /// Identity of the piece the current plan/think is for, to detect a new piece.
    /// `None` before the first piece is seen.
    planning_for: Option<PieceSignature>,
    /// Reaction time accumulated for the current piece (integrates poll `dt`).
    think_elapsed: f32,
    /// Whether a search has been submitted for the current piece (so we submit
    /// once per piece, not every poll while thinking).
    search_submitted: bool,
}

/// A cheap fingerprint of "which piece, on which board" so the controller can tell
/// a genuinely new planning situation from re-polling the same one.
///
/// Piece *type* alone is ambiguous (the next piece can repeat across a bag
/// boundary); pairing it with the locked-cell count distinguishes "same piece,
/// same board" (keep planning) from "new piece spawned" or "board changed under
/// us" (replan). Cheap to compute from a snapshot and `Eq`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PieceSignature {
    piece_type: PieceType,
    /// Number of locked cells on the board (changes on any lock/clear).
    locked_cells: usize,
}

impl PieceSignature {
    /// Derive the signature of a snapshot's current planning situation, if it has
    /// an active piece.
    fn of(snapshot: &EngineSnapshot) -> Option<Self> {
        let piece_type = snapshot.active.as_ref()?.piece_type;
        Some(Self {
            piece_type,
            locked_cells: snapshot.board_cells.len(),
        })
    }
}

impl AiController {
    /// A controller with the shipped Tier-1 stack: greedy planner, linear
    /// evaluator, synchronous runner — and the given difficulty + RNG seed.
    pub fn new(difficulty: DifficultyConfig, seed: u64) -> Self {
        let runner = SyncRunner::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::default()),
        );
        Self::with_runner(Box::new(runner), difficulty, seed)
    }

    /// A controller with the **default** difficulty (a beatable opponent) and the
    /// default AI seed — the convenient construction for the game and sandbox.
    pub fn beatable() -> Self {
        Self::new(DifficultyConfig::default(), DEFAULT_AI_SEED)
    }

    /// A controller around an explicit [`ComputeRunner`], for swapping in an
    /// off-thread / time-sliced runner (or a custom planner+evaluator) without
    /// changing the controller logic.
    pub fn with_runner(
        runner: Box<dyn ComputeRunner>,
        difficulty: DifficultyConfig,
        seed: u64,
    ) -> Self {
        Self {
            runner,
            plan: VecDeque::new(),
            rng: StdRng::seed_from_u64(seed),
            difficulty,
            planning_for: None,
            think_elapsed: 0.0,
            search_submitted: false,
        }
    }

    /// The current difficulty (read-only).
    pub fn difficulty(&self) -> DifficultyConfig {
        self.difficulty
    }

    /// Retune difficulty live (e.g. a slider in the sandbox). Takes effect on the
    /// next piece.
    pub fn set_difficulty(&mut self, difficulty: DifficultyConfig) {
        self.difficulty = difficulty;
    }

    /// The search budget derived from the current difficulty.
    fn budget(&self) -> SearchBudget {
        SearchBudget {
            nodes: self.difficulty.nodes_per_tick,
            max_depth: self.difficulty.max_depth.max(1),
        }
    }

    /// Begin planning for a freshly seen piece: reset the think timer and the
    /// in-flight plan, and remember what we are planning for.
    fn begin_new_piece(&mut self, signature: PieceSignature) {
        self.runner.cancel();
        self.plan.clear();
        self.think_elapsed = 0.0;
        self.search_submitted = false;
        self.planning_for = Some(signature);
    }

    /// Choose which placement to actually play from the planner's best plan,
    /// applying error injection.
    ///
    /// With probability `1 - error_rate` (or always, at `error_rate == 0`) this is
    /// the planner's chosen best. Otherwise it re-runs a quick candidate scan and
    /// softmax-samples among the top-N, so the "mistake" is a plausible near-best
    /// placement rather than nonsense. Uses [`self.rng`](Self) only.
    ///
    /// Returns the placement to render, or the planner's `best` unchanged when
    /// error injection does not fire / cannot improve on it.
    fn choose_placement(&mut self, state: &SearchState, best: PlacementPlan) -> PlacementPlan {
        let rate = self.difficulty.error_rate.clamp(0.0, 1.0);
        if rate <= 0.0 || !self.rng.random_bool(rate as f64) {
            return best;
        }

        // Error fired: gather scored candidates and softmax-sample the top window.
        let mut scored = score_candidates(state);
        if scored.len() <= 1 {
            return best; // nothing to substitute
        }
        // Highest score first. A *stable* sort keeps movegen's canonical candidate
        // order for equal scores, so the sample set is reproducible before the
        // (seeded) softmax draw.
        scored.sort_by_key(|c| std::cmp::Reverse(c.score));
        scored.truncate(DifficultyConfig::ERROR_SAMPLE_WINDOW);

        // Softmax weights over the window, relative to the top score so the
        // exponent stays small and positive. A higher error_rate raises the
        // temperature, flattening the distribution (more likely to pick a worse
        // option).
        let top = scored[0].score;
        let temperature = 1.0 + f64::from(rate) * 8.0; // wider sampling as error_rate rises
        let chosen = scored
            .choose_weighted(&mut self.rng, |c| {
                let delta = f64::from(c.score - top); // <= 0
                (delta / temperature).exp() // in (0, 1]
            })
            .cloned();

        chosen.unwrap_or(best)
    }

    /// Try to apply a ready search result: pick a placement, render it to frames,
    /// and enqueue them. Returns `true` if a plan was applied (frames enqueued or
    /// an explicit "no move" handled), `false` if the runner had nothing yet.
    fn try_apply_plan(&mut self, state: &SearchState) -> bool {
        let Some(result) = self.runner.poll() else {
            return false;
        };
        // A completed search with no legal placement: nothing to do (the engine
        // will top out on its own); leave the plan empty so we emit neutral frames.
        let Some(best) = result else {
            return true;
        };

        let chosen = self.choose_placement(state, best);
        // Render against the board the maneuver happens on, from the active piece's
        // current pose — the same inputs `placement_to_inputs` round-trips.
        let frames = placement_to_inputs(&state.board, &state.active, &chosen.placement);
        self.plan = frames.into();
        true
    }
}

impl PlayerController for AiController {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
        // (1) No active piece (pre-spawn / game over): nothing to plan; idle.
        let Some(signature) = PieceSignature::of(snapshot) else {
            return neutral();
        };

        // (1b) New planning situation? Reset and start a fresh think.
        if self.planning_for != Some(signature) {
            self.begin_new_piece(signature);
        }

        // If we still have queued frames for the current piece, keep draining them
        // (don't re-plan mid-maneuver). One pulse per poll. Maneuver frames carry
        // their own `dt == 0` so positioning advances no gravity — emit as-is.
        if let Some(frame) = self.plan.pop_front() {
            return frame;
        }

        // Build the search state once; reused for submit + error-injection rescan.
        let Some(state) = SearchState::from_snapshot(snapshot) else {
            return neutral();
        };

        // (2) Submit the search once per piece (so it can run during think-time),
        // then accumulate think-time. We integrate think-time at the fixed sim
        // slice (the driver steps at `SIM_HZ`); only the placement *choice* is
        // randomized, so this stays deterministic given the poll cadence.
        if !self.search_submitted {
            self.runner.submit(state.clone(), self.budget());
            self.search_submitted = true;
        }
        self.think_elapsed += NOMINAL_DT;
        if self.think_elapsed < self.difficulty.think_time.as_secs_f32() {
            return neutral(); // still "thinking"
        }

        // (3) Think-time elapsed: apply the plan if the runner is ready, then emit
        // its first maneuver frame.
        if self.try_apply_plan(&state) {
            if let Some(frame) = self.plan.pop_front() {
                return frame;
            }
        }

        // (4) Nothing to emit yet (search not finished, or no legal move): idle.
        neutral()
    }
}

/// Scored candidate placements for error injection — mirrors the greedy planner's
/// scoring but keeps every candidate (the planner only returns the best).
fn score_candidates(state: &SearchState) -> Vec<PlacementPlan> {
    use crate::ai::movegen;
    use crate::engine::{classify_t_spin, lock_and_clear};

    let eval = LinearEvaluator::default();
    let candidates = movegen::generate_with_hold(
        &state.board,
        &state.active,
        state.hold,
        state.queue.front().copied(),
        |piece_type| movegen::spawn_piece(piece_type, state.board.width(), state.board.height()),
    );
    candidates
        .into_iter()
        .map(|placement| {
            let mut board = state.board.clone();
            let t_spin = classify_t_spin(&placement.piece, &board);
            let lock = lock_and_clear(&placement.piece, &mut board);
            let (value, reward) = eval.evaluate(&lock, &board, t_spin);
            PlacementPlan {
                placement,
                score: (value + reward).0,
            }
        })
        .collect()
}

/// Nominal per-poll `dt` for the controller's think-time integration: the fixed
/// sim slice at `SIM_HZ` (60 Hz). The driver steps the engine at that rate, so
/// integrating think-time in these units paces "reaction delay" in real seconds.
/// Maneuver frames carry their *own* `dt == 0` (gravity-free positioning) and are
/// emitted unchanged; only neutral "thinking"/idle frames advance time.
const NOMINAL_DT: f32 = 1.0 / 60.0;

/// A neutral frame: advance one sim slice of time, press nothing. Emitted while
/// thinking or idle so gravity and the lock timer keep ticking between maneuvers.
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
    fn play(controller: &mut AiController, seed: u64, max_frames: usize) -> (usize, bool) {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        let mut locks = 0usize;
        let mut prev_locked = 0usize;
        let mut topped_out = false;
        for _ in 0..max_frames {
            let snapshot = engine.snapshot();
            if snapshot.game_over.is_some() {
                topped_out = true;
                break;
            }
            engine.step(controller.poll(&snapshot));
            // Count locks by watching the locked-cell count step up.
            let now = engine.snapshot().board_cells.len();
            if now > prev_locked {
                locks += 1;
            }
            prev_locked = now;
        }
        (locks, topped_out)
    }

    #[test]
    fn controller_drives_a_real_game_placing_many_pieces() {
        // Driving a real engine through the controller (no dt override — the
        // `drive_engine` contract) must actually play: the controller positions
        // and locks a long run of pieces. (How *well* it plays — line-clear rate,
        // survival depth — is the evaluator's job, exercised in `eval`/`search`
        // tests, not the controller's.)
        let mut controller = AiController::new(DifficultyConfig::perfect(), DEFAULT_AI_SEED);
        let (locks, _topped_out) = play(&mut controller, 7, 4_000);
        assert!(locks >= 20, "the controller should place many pieces, placed {locks}");
    }

    #[test]
    fn controller_executes_the_planners_first_placement_faithfully() {
        // The fidelity guarantee AI3.5 owns: the controller lands its first piece
        // exactly where the *planner* intends. We plan independently from the first
        // snapshot, then drive a perfect (instant, no-error) controller until the
        // piece locks, and assert the resulting board equals the planner's
        // simulated placement. (Plan-to-input pose fidelity is separately pinned by
        // `plan.rs`'s round-trip test; this proves the controller *sequences* it.)
        use crate::ai::search::{GreedyPlanner, Planner, PlannerStep, SearchBudget};
        use crate::ai::SearchState;
        use crate::engine::{lock_and_clear, Board, CellKind};

        let mut controller = AiController::new(DifficultyConfig::perfect(), DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();
        let config = snapshot.config.clone();

        // Planner's intended board after placing the first piece (hold-aware, like
        // the controller's own planner).
        let state = SearchState::from_snapshot(&snapshot).unwrap();
        let plan = match GreedyPlanner::new().plan(&state, &LinearEvaluator::default(), SearchBudget::greedy()) {
            PlannerStep::Done(Some(plan)) => plan,
            other => panic!("expected a plan, got {other:?}"),
        };
        let mut intended = state.board.clone();
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
                    config.buffer_height,
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
    fn neutral_frame_while_thinking() {
        // With a long think-time, the very first polls emit neutral frames (no
        // button pressed) — the bot is "reacting" before it acts.
        let difficulty = DifficultyConfig {
            think_time: core::time::Duration::from_millis(500),
            ..DifficultyConfig::perfect()
        };
        let mut controller = AiController::new(difficulty, DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 1);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();

        let frame = controller.poll(&snapshot);
        assert!(
            !pressed_anything(&frame),
            "first poll under a long think-time must be neutral"
        );
        assert!(frame.dt_seconds > 0.0, "neutral frames still advance time");
    }

    #[test]
    fn determinism_same_seed_same_game() {
        // Two controllers with the same AI seed, driving two engines with the same
        // engine seed, must produce byte-identical games — the determinism the
        // headless benchmark relies on. Uses the default (error-injecting)
        // difficulty so the RNG path is actually exercised.
        let play_to_snapshot = |ai_seed: u64| {
            let mut controller = AiController::new(DifficultyConfig::default(), ai_seed);
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
    fn error_injection_uses_only_the_ai_rng() {
        // A controller with a high error rate must still be fully reproducible from
        // its seed (RNG is owned + seeded, never the engine's). Different AI seeds
        // may diverge; the SAME seed must not.
        let high_error = DifficultyConfig {
            error_rate: 0.9,
            ..DifficultyConfig::perfect()
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
        assert_eq!(run(123), run(123), "same AI seed reproduces even with errors");
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
}
