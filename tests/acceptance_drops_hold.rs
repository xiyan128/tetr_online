//! Acceptance suite for the guideline §25.4 "Drops and Hold".
//!
//! Each scenario is one `#[test]` and is driven exclusively through the public
//! engine boundary (`Engine::new` + `step()` + `snapshot()`), with the sole
//! exception of the two pure-helper scenarios that call the already-public free
//! functions `soft_drop_speed_seconds` / `fall_speed_seconds`.
//!
//! Everything here is deterministic: a fixed seed feeds `StdRng::seed_from_u64`,
//! so the next-queue is reproducible. Where a scenario depends on which piece
//! type spawns, the expected value is read from the snapshot's `next_queue`
//! rather than hard-coded, which keeps the assertions exact without coupling to
//! a particular bag shuffle.

use tetr_online::engine::{
    fall_speed_seconds, soft_drop_speed_seconds, Engine, EngineConfig, EngineEvent,
    EngineScoreAction, InputFrame, PieceRotation,
};

/// Fixed seed shared by every engine-driven scenario for determinism.
const SEED: u64 = 0;

/// Builds an engine on the default config (10-wide board) at the fixed seed.
fn default_engine() -> Engine {
    Engine::new(EngineConfig::default(), SEED)
}

/// Spawns the first active piece via a zero-input step and returns the events.
fn spawn_first_piece(engine: &mut Engine) -> Vec<EngineEvent> {
    engine.step(InputFrame::default())
}

// §25.4 / §6.5 / §9.2 — Soft Drop fall speed is exactly normalFallSpeed / 20.
#[test]
fn soft_drop_speed_is_twenty_times_fall_speed() {
    for level in [1u8, 15u8] {
        assert_eq!(
            soft_drop_speed_seconds(level),
            fall_speed_seconds(level) / 20.0,
            "soft drop speed must be 1/20th of normal fall speed at level {level}",
        );
    }
}

// §25.4 / §6.5 / §13.1 — Soft Drop awards 1 point per row descended.
#[test]
fn soft_drop_scores_one_per_descended_row() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);
    let before = engine.snapshot().active.expect("active piece after spawn");
    let expected_origin = (before.origin.0, before.origin.1 - 1);

    let events = engine.step(InputFrame {
        soft_drop: true,
        ..InputFrame::default()
    });

    // The descent itself is snapshot state (origin down exactly one row); the
    // event stream carries only the 1-point soft-drop award.
    assert_eq!(
        events,
        vec![EngineEvent::ScoreAwarded {
            action: EngineScoreAction::SoftDrop,
            score: 1,
            total_score: 1,
            back_to_back_bonus: false,
        }],
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after soft drop")
            .origin,
        expected_origin,
        "one soft-drop frame descends exactly one row",
    );
    assert_eq!(engine.snapshot().score, 1);
}

// §25.4 / §6.4 — Hard Drop drops to the surface, scores, locks immediately, and
// the next piece spawns. On an empty 10-wide board exactly one piece (4 minos)
// is locked and no lines clear.
#[test]
fn hard_drop_lands_and_locks_immediately() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);
    let piece_type = engine
        .snapshot()
        .active
        .expect("active piece after spawn")
        .piece_type;
    let next_type = engine.snapshot().next_queue[0];

    let events = engine.step(InputFrame {
        hard_drop: true,
        ..InputFrame::default()
    });

    // `cells_dropped` is determined by the spawn height; capture it from the
    // first event so the full-vector comparison stays exact yet deterministic.
    let cells = match events.first() {
        Some(EngineEvent::HardDropped { cells_dropped, .. }) => *cells_dropped,
        other => panic!("expected HardDropped first, got {other:?}"),
    };
    assert!(cells > 0, "first piece must descend at least one row");

    assert_eq!(
        events,
        vec![
            EngineEvent::HardDropped {
                piece_type,
                cells_dropped: cells,
            },
            EngineEvent::ScoreAwarded {
                action: EngineScoreAction::HardDrop { cells },
                score: 2 * cells,
                total_score: 2 * cells,
                back_to_back_bonus: false,
            },
            EngineEvent::Locked {
                piece_type,
                lines_cleared: 0,
            },
        ],
    );

    assert_eq!(
        engine.snapshot().board_cells.len(),
        4,
        "exactly one tetromino (4 minos) should be locked to the board",
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("the locking step spawns the next piece")
            .piece_type,
        next_type,
        "the same step that locks must leave the next queued piece active",
    );
}

// §25.4 / §6.4 / §13.1 — Hard Drop awards 2 points per row descended; the
// awarded score equals 2 * HardDropped.cells_dropped.
#[test]
fn hard_drop_scores_two_per_descended_row() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);

    let events = engine.step(InputFrame {
        hard_drop: true,
        ..InputFrame::default()
    });

    let cells_dropped = match events.iter().find_map(|event| match event {
        EngineEvent::HardDropped { cells_dropped, .. } => Some(*cells_dropped),
        _ => None,
    }) {
        Some(cells) => cells,
        None => panic!("expected a HardDropped event in {events:?}"),
    };

    let (action_cells, score) = match events.iter().find_map(|event| match event {
        EngineEvent::ScoreAwarded {
            action: EngineScoreAction::HardDrop { cells },
            score,
            ..
        } => Some((*cells, *score)),
        _ => None,
    }) {
        Some(pair) => pair,
        None => panic!("expected a HardDrop ScoreAwarded event in {events:?}"),
    };

    assert_eq!(
        action_cells, cells_dropped,
        "ScoreAwarded cells must equal HardDropped.cells_dropped",
    );
    assert_eq!(
        score,
        2 * cells_dropped,
        "Hard Drop must award 2 points per descended row",
    );
}

// §25.4 / §6.6 (empty hold branch) — Holding with an empty Hold queue stores the
// active piece and spawns the next one from the Next queue.
#[test]
fn hold_with_empty_hold_stores_active_and_spawns_next() {
    let mut engine = default_engine();
    let initial_queue = engine.snapshot().next_queue;
    let first_piece_type = initial_queue[0];
    let next_piece_type = initial_queue[1];
    spawn_first_piece(&mut engine);

    let events = engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });

    // The hold's event is the swap itself; the incoming spawn is snapshot
    // state, pinned by the `active` assertion below.
    assert_eq!(
        events,
        vec![EngineEvent::Held {
            held: first_piece_type,
            active: next_piece_type,
        }],
    );

    let snapshot = engine.snapshot();
    assert_eq!(snapshot.hold, Some(first_piece_type));
    assert_eq!(
        snapshot.active.expect("active piece after hold").piece_type,
        next_piece_type,
    );
}

// §25.4 / §6.6 step 4 — A held piece respawns North Facing (rotation R0).
#[test]
fn held_piece_respawns_north_facing() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);

    engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });

    let active = engine.snapshot().active.expect("active piece after hold");
    assert_eq!(
        active.rotation,
        PieceRotation::R0,
        "swapped-in piece must spawn North Facing (R0)",
    );
}

// §25.4 / §6.6 step 2 / §3.3 — Hold may be used only once per active piece; a
// Lock Down must occur between Holds. The second hold on the same piece is a
// no-op (no events, snapshot unchanged).
#[test]
fn cannot_hold_twice_before_lock_down() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);
    engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });
    let before = engine.snapshot();

    let events = engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });

    assert!(
        events.is_empty(),
        "second hold on the same piece must emit no events, got {events:?}",
    );
    assert_eq!(
        engine.snapshot(),
        before,
        "second hold on the same piece must not change engine state",
    );
}

// §25.4 / §6.6 step 4 — Holding when the Hold queue already contains a piece
// swaps the active piece with the held one, respawning North Facing.
//
// Reached purely through inputs across two pieces (no private seam):
//   1. spawn piece A, hold it     -> Hold queue = A, active = B (hold_used)
//   2. hard-drop B to lock it     -> spawns piece C with a fresh hold permission
//   3. hold on C                  -> swaps C out, A back in (the existing held)
// The swapped-in piece must equal the previously held piece A and face R0.
#[test]
fn hold_swaps_with_existing_held_piece() {
    let mut engine = default_engine();
    spawn_first_piece(&mut engine);

    // Step 1: store piece A into the (empty) Hold queue.
    engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });
    let previously_held = engine
        .snapshot()
        .hold
        .expect("hold queue holds piece A after first hold");

    // Step 2: lock the current active piece so the next spawn regains its
    // per-piece hold permission. A hard drop on the empty 10-wide well locks a
    // single piece and clears no lines, so the engine keeps running.
    let drop_events = engine.step(InputFrame {
        hard_drop: true,
        ..InputFrame::default()
    });
    assert!(
        engine.snapshot().active.is_some(),
        "locking the active piece must spawn the next piece, got {drop_events:?}",
    );
    assert!(
        engine.snapshot().game_over.is_none(),
        "hard drop into an empty well must not end the game",
    );

    // Snapshot the swap inputs/outputs around the second hold.
    let active_before_swap = engine
        .snapshot()
        .active
        .expect("active piece C before the swap")
        .piece_type;

    // Step 3: hold again — this swaps with the existing held piece.
    let swap_events = engine.step(InputFrame {
        hold: true,
        ..InputFrame::default()
    });

    let snapshot = engine.snapshot();
    let active_after = snapshot.active.expect("active piece after swap");

    // The swapped-in piece is the piece that was previously in the Hold queue,
    // and the active piece that was swapped out now occupies the Hold queue.
    assert_eq!(
        active_after.piece_type, previously_held,
        "the incoming piece must equal the previously held piece",
    );
    assert_eq!(
        snapshot.hold,
        Some(active_before_swap),
        "the outgoing active piece must now occupy the Hold queue",
    );
    assert_eq!(
        active_after.rotation,
        PieceRotation::R0,
        "the swapped-in piece must respawn North Facing (R0)",
    );

    // The swap emits a Held event recording both sides of the exchange (the
    // incoming piece's spawn is the snapshot state asserted above).
    assert!(
        swap_events.contains(&EngineEvent::Held {
            held: active_before_swap,
            active: previously_held,
        }),
        "expected a Held event swapping {active_before_swap:?} out for {previously_held:?}, got {swap_events:?}",
    );
}
