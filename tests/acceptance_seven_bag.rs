//! Acceptance tests for the guideline §25.2 "Seven-bag and queues".
//!
//! Spec bullets covered:
//!   * Every bag of seven contains exactly one of each piece.
//!   * Bag refills only when empty (two consecutive bags each complete).
//!   * Next Queue shifts correctly after generation.
//!   * Visible Next Queue count supports 1..6.
//!   * Same seed + same config is deterministic (§3.1 "Use a seeded RNG in tests").
//!
//! All tests drive the engine exclusively through its public boundary
//! (`Engine::new` / `step` / `snapshot`) with a fixed seed, so they are fully
//! deterministic. The engine's spawn sequence is the order in which pieces are
//! popped off the front of the Next Queue, so the queue contents (read via
//! `snapshot().next_queue`) are the upcoming spawns in order — this is the
//! "next_queue draining" path the spec allows for collecting spawned pieces.

use tetr_online::{Engine, EngineConfig, EngineEvent, EngineSnapshot, InputFrame, PieceType};

/// Fixed seed so every assertion below is reproducible.
const SEED: u64 = 0xACCE_57ED;

/// `PieceType` is `Eq` but not `Ord`, so canonicalize a multiset of pieces by
/// sorting on the enum discriminant. `PieceType::ALL` is declared in
/// discriminant order (`I, J, L, O, S, T, Z`), so a sorted full bag equals it.
fn sorted_by_discriminant(pieces: &[PieceType]) -> Vec<PieceType> {
    let mut sorted = pieces.to_vec();
    sorted.sort_by_key(|piece| *piece as u8);
    sorted
}

/// Run one frame with all inputs cleared (zero dt). With no active piece this
/// spawns the next piece (and applies its immediate one-row drop).
fn idle_frame() -> InputFrame {
    InputFrame::default()
}

/// Run one frame that hard-drops the active piece, locking it immediately and
/// clearing `active` so the following idle frame spawns the next piece.
fn hard_drop_frame() -> InputFrame {
    InputFrame {
        hard_drop: true,
        ..InputFrame::default()
    }
}

/// Extract the piece type from a `Spawned` event, if present.
fn spawned_piece(events: &[EngineEvent]) -> Option<PieceType> {
    events.iter().find_map(|event| match event {
        EngineEvent::Spawned { piece_type } => Some(*piece_type),
        _ => None,
    })
}

#[test]
fn each_bag_of_seven_contains_one_of_each_piece() {
    // A preview of 7 surfaces exactly the first dealt bag as the Next Queue,
    // so we can read the seven upcoming spawns without ever stacking the board.
    let config = EngineConfig {
        preview_count: 7,
        ..Default::default()
    };
    let engine = Engine::new(config, SEED);

    let bag = engine.snapshot().next_queue;
    assert_eq!(
        bag.len(),
        7,
        "preview_count = 7 must surface seven queued pieces",
    );

    // §25.2: every bag of seven contains exactly one of each piece.
    assert_eq!(
        sorted_by_discriminant(&bag),
        PieceType::ALL.to_vec(),
        "a single bag must be a permutation of all seven piece types, got {bag:?}",
    );
}

#[test]
fn two_consecutive_bags_each_complete() {
    // 14 previewed pieces = the first two dealt bags, in spawn order. Because
    // the bag refills only when empty, the boundary falls exactly at index 7.
    let config = EngineConfig {
        preview_count: 14,
        ..Default::default()
    };
    let engine = Engine::new(config, SEED);

    let pieces = engine.snapshot().next_queue;
    assert_eq!(
        pieces.len(),
        14,
        "preview_count = 14 must surface fourteen queued pieces",
    );

    // §25.2: bag refills only when empty -> each disjoint 7-window is a full,
    // non-overlapping permutation of all seven piece types.
    for (bag_index, window) in pieces.chunks_exact(7).enumerate() {
        assert_eq!(
            sorted_by_discriminant(window),
            PieceType::ALL.to_vec(),
            "bag #{bag_index} must contain exactly one of each piece, got {window:?}",
        );
    }
}

#[test]
fn next_queue_shifts_after_each_spawn() {
    // Default preview (5). Spawn the first piece, then lock it and spawn the
    // second, and verify the queue advanced by exactly one each spawn while its
    // length stays pinned to preview_count.
    let config = EngineConfig::default();
    let preview_count = config.preview_count;
    let mut engine = Engine::new(config, SEED);

    // Snapshot the queue before any spawn. front = first piece to enter.
    let queue_before_first = engine.snapshot().next_queue;
    assert_eq!(queue_before_first.len(), preview_count);
    let first_piece = queue_before_first[0];

    // First step spawns the front piece and refills the queue from the back.
    let first_events = engine.step(idle_frame());
    assert_eq!(
        spawned_piece(&first_events),
        Some(first_piece),
        "first spawn must consume the front of the Next Queue",
    );

    let queue_after_first_spawn = engine.snapshot().next_queue;
    // The front was consumed: the queue advanced by one, so its new front is the
    // old second element, and the old front is gone from the head.
    assert_eq!(
        queue_after_first_spawn[0], queue_before_first[1],
        "queue must advance so the previous second piece becomes the new front",
    );
    assert_eq!(
        queue_after_first_spawn.len(),
        preview_count,
        "queue length must stay pinned to preview_count after a spawn",
    );

    // The piece that will spawn next is now at the front of the queue.
    let expected_second_piece = queue_after_first_spawn[0];

    // Hard-dropping locks piece 1 and, in the same step, spawns piece 2: the
    // engine's Completion -> Generation transition emits `Locked` then `Spawned`
    // within one `step()` (the same contract asserted by
    // `acceptance_drops_hold::hard_drop_lands_and_locks_immediately` and the
    // lock-timer-expiry cases in `acceptance_lock_down`). So the second spawn is
    // observed in the hard-drop step's events, not a following idle frame.
    let second_events = engine.step(hard_drop_frame());
    assert_eq!(
        spawned_piece(&second_events),
        Some(expected_second_piece),
        "second spawn must consume the new front of the Next Queue",
    );

    let queue_after_second_spawn = engine.snapshot().next_queue;
    assert_eq!(
        queue_after_second_spawn[0], queue_after_first_spawn[1],
        "queue must advance again so the next piece becomes the front",
    );
    assert_eq!(
        queue_after_second_spawn.len(),
        preview_count,
        "queue length must remain preview_count across successive spawns",
    );
}

#[test]
fn visible_next_queue_count_supports_1_through_6() {
    // §25.2 / §3.2: legal visible Next Queue range is 1 through 6.
    for preview_count in 1..=6 {
        let config = EngineConfig {
            preview_count,
            ..Default::default()
        };
        let engine = Engine::new(config, SEED);

        assert_eq!(
            engine.snapshot().next_queue.len(),
            preview_count,
            "Next Queue must surface exactly {preview_count} previewed pieces",
        );
    }
}

#[test]
fn same_seed_same_config_is_deterministic() {
    // §3.1: a seeded RNG must reproduce the same sequence. Two engines built
    // with the same (config, seed) and fed identical input sequences must hold
    // identical snapshots at every observation point.
    let config = EngineConfig::default();
    let mut left = Engine::new(config.clone(), SEED);
    let mut right = Engine::new(config, SEED);

    // Equal before any input (initial preview queue must match).
    assert_eq!(
        left.snapshot(),
        right.snapshot(),
        "fresh engines with the same (config, seed) must start identical",
    );

    // An identical, varied input sequence covering spawn, move, rotate, soft
    // drop, hold, and several hard-drop/respawn cycles.
    let input_sequence = [
        idle_frame(),
        InputFrame {
            left: true,
            ..InputFrame::default()
        },
        InputFrame {
            rotate_clockwise: true,
            ..InputFrame::default()
        },
        InputFrame {
            soft_drop: true,
            dt_seconds: 0.05,
            ..InputFrame::default()
        },
        InputFrame {
            hold: true,
            ..InputFrame::default()
        },
        hard_drop_frame(),
        idle_frame(),
        InputFrame {
            right: true,
            ..InputFrame::default()
        },
        hard_drop_frame(),
        idle_frame(),
    ];

    for (frame_index, input) in input_sequence.iter().enumerate() {
        let left_events = left.step(input.clone());
        let right_events = right.step(input.clone());
        assert_eq!(
            left_events, right_events,
            "events must match at frame {frame_index} for identical (config, seed, input)",
        );

        let left_snapshot: EngineSnapshot = left.snapshot();
        let right_snapshot: EngineSnapshot = right.snapshot();
        assert_eq!(
            left_snapshot, right_snapshot,
            "snapshots must match at frame {frame_index} for identical (config, seed, input)",
        );
    }
}
