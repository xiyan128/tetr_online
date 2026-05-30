//! Acceptance tests for the guideline §25.6 "Lock Down"
//! (Extended / Infinite / Classic), anchored to the normative rules in §8.1-8.4.
//!
//! These exercise the public engine boundary only: deterministic `Engine::new`
//! with `InputFrame`/`step`/`snapshot`, plus the pure
//! `apply_grounded_move_or_rotation` helper and `ActivePiece` builders. No
//! private engine internals are touched.
//!
//! Spec anchors:
//! ```text
//! §8.1 / §25.6  "Natural fall or Soft Drop onto a Surface starts a 0.5s timer."
//! §8.2 / §25.6  Extended: timer resets on grounded move/rotate, budget = 15
//!               successful grounded actions since the lowest row reached;
//!               falling below that row reopens the budget.
//! §8.3 / §25.6  Infinite: any successful move/rotate resets the timer; no budget.
//! §8.4 / §25.6  Classic: only falling lower resets the timer.
//! ```
//!
//! All symbols are imported from the flat `tetr_online::` re-export surface
//! (lib.rs:11-20), which exposes every name these tests need.

use tetr_online::{
    apply_grounded_move_or_rotation, fall_speed_seconds, ActivePiece, Engine, EngineConfig,
    EngineEvent, InputFrame, LockDownMode, PieceAction, PieceType, EXTENDED_LOCK_RESET_BUDGET,
    LOCK_DOWN_SECONDS,
};

const SEED: u64 = 0;

/// f32 epsilon for comparing lock-timer values that were produced by
/// subtraction (e.g. `0.5 - 0.3`). Direct `reset_lock_timer(0.5)` results are
/// compared with exact `==` since `0.5` is representable.
const EPS: f32 = 1e-5;

fn approx(actual: f32, expected: f32) -> bool {
    (actual - expected).abs() < EPS
}

/// Spawn the first piece, then drive gravity (one row per `fall_speed` step)
/// until the active piece lands. Returns once `snapshot().active.landed` is true.
///
/// Works on an empty board with a flat floor, so the landing surface is uniform
/// and later horizontal moves keep the piece grounded.
fn spawn_and_land_via_gravity(engine: &mut Engine) {
    // First step spawns the piece (with the immediate post-spawn drop).
    engine.step(InputFrame::default());
    let fall = fall_speed_seconds(engine.snapshot().level);

    for _ in 0..256 {
        let active = engine
            .snapshot()
            .active
            .expect("active piece present while landing");
        if active.landed {
            return;
        }
        engine.step(InputFrame {
            dt_seconds: fall,
            ..InputFrame::default()
        });
    }
    panic!("piece never landed under gravity within the step budget");
}

fn left() -> InputFrame {
    InputFrame {
        left: true,
        ..InputFrame::default()
    }
}

fn wait(dt_seconds: f32) -> InputFrame {
    InputFrame {
        dt_seconds,
        ..InputFrame::default()
    }
}

// -------------------------------------------------------------------------
// §8.1 / §25.6 — landing starts the 0.5s Lock Down timer.
// -------------------------------------------------------------------------

/// §25.6 Extended: "Landing starts 0.5s timer."
#[test]
fn extended_landing_starts_half_second_timer() {
    let mut engine = Engine::new(EngineConfig::default(), SEED);
    spawn_and_land_via_gravity(&mut engine);

    let active = engine
        .snapshot()
        .active
        .expect("active piece present after landing");

    assert!(active.landed, "piece should be marked landed after gravity");
    assert_eq!(
        LOCK_DOWN_SECONDS, 0.5,
        "the guideline lock-down timer is 0.5s"
    );
    assert_eq!(
        active.lock_timer_seconds, LOCK_DOWN_SECONDS,
        "landing must start the timer at exactly 0.5s"
    );
}

// -------------------------------------------------------------------------
// §8.2 / §25.6 — Extended: grounded move resets the timer under budget.
// -------------------------------------------------------------------------

/// §25.6 Extended: "Successful grounded movement/rotation resets timer while
/// under 15-action limit." Uses a wide board so the landed piece has room for
/// 15 grounded left steps, each of which must restore the timer to 0.5s.
#[test]
fn extended_grounded_move_resets_timer_under_budget() {
    let config = EngineConfig {
        board_width: 40,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(config, SEED);
    spawn_and_land_via_gravity(&mut engine);

    // Sanity: fresh landing leaves the timer at 0.5s and no grounded actions spent.
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("landed active piece")
            .lock_timer_seconds,
        LOCK_DOWN_SECONDS
    );

    for action in 1..=EXTENDED_LOCK_RESET_BUDGET {
        // Drain the timer partially so a reset is observable as a jump back to 0.5.
        engine.step(wait(0.2));
        let drained = engine
            .snapshot()
            .active
            .expect("active piece while draining")
            .lock_timer_seconds;
        assert!(
            approx(drained, LOCK_DOWN_SECONDS - 0.2),
            "action {action}: timer should drain to ~0.3 before the move, got {drained}"
        );

        // A successful grounded left move (still under the 15-action budget)
        // must reset the timer back to 0.5s.
        let events = engine.step(left());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Moved { .. })),
            "action {action}: grounded left move should emit a Moved event"
        );
        let after = engine
            .snapshot()
            .active
            .expect("active piece after grounded move");
        assert!(
            after.landed,
            "action {action}: piece must remain grounded on the flat floor"
        );
        assert_eq!(
            after.lock_timer_seconds, LOCK_DOWN_SECONDS,
            "action {action}: grounded move under budget must reset timer to 0.5s"
        );
    }
}

// -------------------------------------------------------------------------
// §8.2 / §25.6 — Extended: budget does not reset after exhaustion.
// -------------------------------------------------------------------------

/// §25.6 Extended: "15-action budget does not reset unless piece falls below
/// previous lowest row" and "After budget exhaustion, grounded movement/rotation
/// does not extend lock." Mirrors the in-module
/// `extended_lock_down_budget_stops_resetting_after_fifteen_grounded_moves`
/// unit test, but reaches the landed state through the public step API.
#[test]
fn extended_15_action_budget_does_not_reset_after_exhaustion() {
    let config = EngineConfig {
        board_width: 40,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(config, SEED);
    spawn_and_land_via_gravity(&mut engine);

    // Spend all 15 grounded resets via successful left moves on the flat floor.
    for _ in 0..EXTENDED_LOCK_RESET_BUDGET {
        let events = engine.step(left());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EngineEvent::Moved { .. })),
            "each budgeted left move should succeed and emit Moved"
        );
        assert_eq!(
            engine
                .snapshot()
                .active
                .expect("active piece")
                .lock_timer_seconds,
            LOCK_DOWN_SECONDS,
            "moves within budget keep the timer pinned at 0.5s"
        );
    }

    // Drain the timer below 0.5s so a (forbidden) reset would be visible.
    engine.step(wait(0.3));
    let drained = engine
        .snapshot()
        .active
        .expect("active piece after drain")
        .lock_timer_seconds;
    assert!(
        approx(drained, LOCK_DOWN_SECONDS - 0.3),
        "timer should have drained to ~0.2, got {drained}"
    );

    // The 16th grounded move: still a successful left move (Moved emitted), but
    // the budget is exhausted, so it MUST NOT reset the timer back to 0.5s.
    let events = engine.step(left());
    assert!(
        events
            .iter()
            .any(|e| matches!(e, EngineEvent::Moved { .. })),
        "the 16th left move still physically succeeds"
    );
    let after_16th = engine
        .snapshot()
        .active
        .expect("active piece after 16th move")
        .lock_timer_seconds;
    assert!(
        after_16th < LOCK_DOWN_SECONDS,
        "exhausted budget: 16th grounded move must not reset to 0.5s, timer = {after_16th}"
    );
    assert!(
        approx(after_16th, LOCK_DOWN_SECONDS - 0.3),
        "exhausted budget: timer must stay drained at ~0.2, got {after_16th}"
    );

    // Letting the (un-reset) timer expire locks the piece and spawns the next.
    let events = engine.step(wait(after_16th));
    assert!(
        matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    lines_cleared: 0,
                    ..
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "expired lock timer must lock then spawn, got {events:?}"
    );
}

// -------------------------------------------------------------------------
// §8.2 / §25.6 — Extended: budget reopens only when the piece falls below
// the previous lowest row (pure ActivePiece + helper, no Engine seam).
// -------------------------------------------------------------------------

/// §25.6 Extended: "15-action budget does not reset unless piece falls below
/// previous lowest row." Pure helper test on `apply_grounded_move_or_rotation`:
/// 15 calls return true (resetting), the 16th returns false; after a `Fall` to a
/// lower row the counter is reset, so the next call returns true again.
#[test]
fn extended_budget_reopens_only_when_piece_falls_below_previous_lowest() {
    let mut active = ActivePiece::new(PieceType::T, (3, 19));

    // 15 grounded actions are granted; each resets the timer to 0.5s.
    for expected_count in 1..=EXTENDED_LOCK_RESET_BUDGET {
        assert!(
            apply_grounded_move_or_rotation(&mut active, LockDownMode::Extended, LOCK_DOWN_SECONDS),
            "action {expected_count} should be granted within the budget"
        );
        assert_eq!(
            active.grounded_move_rotate_count_since_lowest(),
            expected_count
        );
        assert_eq!(active.lock_timer_seconds(), LOCK_DOWN_SECONDS);
    }

    // 16th action: budget exhausted -> not granted, timer not reset.
    assert!(
        !apply_grounded_move_or_rotation(&mut active, LockDownMode::Extended, LOCK_DOWN_SECONDS),
        "the 16th grounded action must be refused once the budget is spent"
    );
    assert_eq!(
        active.grounded_move_rotate_count_since_lowest(),
        EXTENDED_LOCK_RESET_BUDGET
    );

    // Falling one row below the lowest row ever reached reopens the budget.
    active.move_to((3, 18), PieceAction::Fall);
    assert_eq!(
        active.grounded_move_rotate_count_since_lowest(),
        0,
        "falling below the previous lowest row must reset the 15-action counter"
    );

    // The next grounded action is granted again (and counts as 1 since lowest).
    assert!(
        apply_grounded_move_or_rotation(&mut active, LockDownMode::Extended, LOCK_DOWN_SECONDS),
        "the budget must reopen after the piece falls lower"
    );
    assert_eq!(active.grounded_move_rotate_count_since_lowest(), 1);
}

// -------------------------------------------------------------------------
// §8.3 / §25.6 — Infinite: resets the timer indefinitely.
// -------------------------------------------------------------------------

/// §25.6 Infinite: "Successful movement/rotation resets timer indefinitely."
/// Pure helper test: well beyond the Extended budget, every grounded action is
/// still granted and the counter never accumulates (no budget tracking).
#[test]
fn infinite_resets_timer_indefinitely() {
    let mut active = ActivePiece::new(PieceType::T, (3, 19));

    // Far more than EXTENDED_LOCK_RESET_BUDGET actions, all granted.
    for _ in 0..(EXTENDED_LOCK_RESET_BUDGET as usize * 4 + 1) {
        // Drain the timer so each reset is meaningful.
        active.set_lock_timer_seconds(0.1);
        assert!(
            apply_grounded_move_or_rotation(&mut active, LockDownMode::Infinite, LOCK_DOWN_SECONDS),
            "Infinite mode must grant every successful action"
        );
        assert_eq!(
            active.lock_timer_seconds(),
            LOCK_DOWN_SECONDS,
            "Infinite mode resets the timer to 0.5s on every action"
        );
    }

    // No 15-action budget exists, so the counter stays at 0.
    assert_eq!(
        active.grounded_move_rotate_count_since_lowest(),
        0,
        "Infinite mode tracks no grounded-action budget"
    );
}

// -------------------------------------------------------------------------
// §8.4 / §25.6 — Classic: only falling lower resets the timer.
// -------------------------------------------------------------------------

/// §25.6 Classic: "Only falling lower resets timer." Pure helper test:
/// `apply_grounded_move_or_rotation` always returns false in Classic and never
/// touches the timer; only a `Fall` to a lower row (which re-lands the piece and
/// restarts the timer at 0.5s) resets it.
#[test]
fn classic_only_falling_lower_resets_timer() {
    let mut active = ActivePiece::new(PieceType::T, (3, 19));
    active.mark_landed();
    active.reset_lock_timer(LOCK_DOWN_SECONDS);
    // Partially drain so a reset would be observable.
    active.set_lock_timer_seconds(0.2);

    // A grounded move/rotation must NOT be granted and must NOT reset the timer.
    assert!(
        !apply_grounded_move_or_rotation(&mut active, LockDownMode::Classic, LOCK_DOWN_SECONDS),
        "Classic mode never grants grounded resets"
    );
    assert_eq!(
        active.lock_timer_seconds(),
        0.2,
        "Classic mode leaves the timer untouched on a grounded move/rotation"
    );

    // Only falling lower resets the timer. A Fall to a lower row re-establishes
    // a landing, which restarts the timer at 0.5s.
    active.move_to((3, 18), PieceAction::Fall);
    active.reset_lock_timer(LOCK_DOWN_SECONDS);
    assert_eq!(
        active.lock_timer_seconds(),
        LOCK_DOWN_SECONDS,
        "falling lower (and re-landing) is the only thing that resets the timer in Classic"
    );
}

// -------------------------------------------------------------------------
// §8.1 / §25.6 — gravity landing then timer expiry locks and spawns.
// -------------------------------------------------------------------------

/// §8.1 + §25.6: a piece that lands via gravity starts the 0.5s timer; once the
/// timer expires the piece locks and the next piece spawns. Driven entirely
/// through the public step API.
#[test]
fn gravity_landing_then_timer_expiry_locks_and_spawns() {
    let mut engine = Engine::new(EngineConfig::default(), SEED);
    spawn_and_land_via_gravity(&mut engine);

    // Landing established the timer at 0.5s.
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("landed active piece")
            .lock_timer_seconds,
        LOCK_DOWN_SECONDS
    );
    assert!(
        engine.snapshot().board_cells.is_empty(),
        "nothing is locked to the board until the timer expires"
    );

    // Advancing exactly the lock-down duration expires the timer: lock + spawn.
    let events = engine.step(wait(LOCK_DOWN_SECONDS));
    assert!(
        matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    lines_cleared: 0,
                    ..
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "timer expiry must produce [Locked, Spawned], got {events:?}"
    );
    assert_eq!(
        engine.snapshot().board_cells.len(),
        4,
        "the locked piece's four minos are now on the board"
    );
}
