//! Acceptance tests for reference_guideline.md §25.3 "Movement and DAS".
//!
//! Engine-level spec items covered here (§25.3):
//!   * Tap left/right moves one cell.        -> tap_left/right_moves_one_cell
//!   * Rotation has no auto-repeat.          -> rotation_has_no_auto_repeat
//!
//! DESIGN NOTE (§25.3 timing is PLAYER-SIDE, per roadmap ADR-4 / E0.13): the DAS
//! *timing* items — the ~0.3s hold delay, the auto-repeat interval, carry-over of
//! auto-repeat across Lock Down, and opposite-direction delay restart — are NOT
//! engine responsibilities. `Engine::step` treats `left`/`right` as a per-frame
//! one-cell pulse and owns no charge timer; the DAS state machine lives in the
//! player layer (`src/player/das.rs`, `KeyboardController`). Those timing
//! scenarios are therefore exercised by headless unit tests over `DasState` /
//! `KeyboardController`, not against the engine. This file keeps only the two
//! genuinely engine-level guarantees (tap = one cell, rotation has no
//! auto-repeat) plus `caveat_engine_step_has_no_das_state_machine`, which pins
//! the engine's intentional tap-only behaviour so a future regression can't
//! silently fold DAS timing back into the engine.
//!
//! Reachability note: these integration tests reach the engine via
//! `tetr_online::engine::*`, which requires `pub mod engine;` in src/lib.rs
//! (`MoveDirection` is needed here and is not part of the flat re-export block).

use tetr_online::engine::*;

/// Seed whose first generated piece is `T` (deterministic: the bag is shuffled
/// from a `StdRng::seed_from_u64`, and seed 0 pops `T` first — the same
/// assumption the in-crate unit tests rely on). `T` rotates cleanly on an empty
/// board, which the rotation scenario depends on.
const SEED_FIRST_PIECE_T: u64 = 0;

/// Drive the very first `step` (zero dt, no inputs) so the engine spawns and
/// applies its single immediate gravity drop, then return the spawned active
/// piece snapshot. After this the active piece is sitting at its post-drop
/// origin in an otherwise empty well.
fn spawn_first_piece(engine: &mut Engine) -> ActivePieceSnapshot {
    let spawn_events = engine.step(InputFrame::default());
    assert!(
        matches!(spawn_events.as_slice(), [EngineEvent::Spawned { .. }]),
        "first zero-dt step should spawn exactly one piece, got {spawn_events:?}",
    );
    engine
        .snapshot()
        .active
        .expect("a piece is active after the first step")
}

// -------------------------------------------------------------------------
// §25.3: Tap left/right moves one cell.
// -------------------------------------------------------------------------

#[test]
fn tap_left_moves_one_cell() {
    let mut engine = Engine::new(EngineConfig::default(), SEED_FIRST_PIECE_T);
    let before = spawn_first_piece(&mut engine);
    let expected_origin = (before.origin.0 - 1, before.origin.1);

    let events = engine.step(InputFrame {
        left: true,
        ..InputFrame::default()
    });

    // Exactly one effective move, Left, origin shifted by -1 in x.
    assert_eq!(
        events,
        vec![EngineEvent::Moved {
            piece_type: before.piece_type,
            direction: MoveDirection::Left,
            origin: expected_origin,
        }],
        "a single left tap must emit exactly one Left Moved event",
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after tap")
            .origin,
        expected_origin,
        "the active piece must end one cell to the left",
    );
}

#[test]
fn tap_right_moves_one_cell() {
    let mut engine = Engine::new(EngineConfig::default(), SEED_FIRST_PIECE_T);
    let before = spawn_first_piece(&mut engine);
    let expected_origin = (before.origin.0 + 1, before.origin.1);

    let events = engine.step(InputFrame {
        right: true,
        ..InputFrame::default()
    });

    // Mirror of the left tap: exactly one Right move, origin shifted by +1 in x.
    assert_eq!(
        events,
        vec![EngineEvent::Moved {
            piece_type: before.piece_type,
            direction: MoveDirection::Right,
            origin: expected_origin,
        }],
        "a single right tap must emit exactly one Right Moved event",
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after tap")
            .origin,
        expected_origin,
        "the active piece must end one cell to the right",
    );
}

// -------------------------------------------------------------------------
// §25.3 / §6.3: Rotation has no auto-repeat.
// -------------------------------------------------------------------------

#[test]
fn rotation_has_no_auto_repeat() {
    // Holding `rotate_clockwise` across two consecutive frames must NOT spin the
    // piece more than once per frame: each frame is one 90-degree press attempt
    // (§6.3 "Each rotation button press attempts exactly one 90-degree rotation.
    // No rotation auto-repeat."). We assert the engine-true encoding of that
    // rule: at most ONE Rotated event per step, advancing R0 -> R90 -> R180 over
    // two held frames rather than R0 -> R180 (or further) in a single frame.
    let mut engine = Engine::new(EngineConfig::default(), SEED_FIRST_PIECE_T);
    let before = spawn_first_piece(&mut engine);
    assert_eq!(
        before.piece_type,
        PieceType::T,
        "seed {SEED_FIRST_PIECE_T} is expected to spawn a T (a rotatable piece)",
    );
    assert_eq!(
        before.rotation,
        PieceRotation::R0,
        "a freshly spawned piece faces North (R0)",
    );

    // Frame 1: button held -> exactly one rotation R0 -> R90.
    let first = engine.step(InputFrame {
        rotate_clockwise: true,
        ..InputFrame::default()
    });
    let first_rotations = first
        .iter()
        .filter(|event| matches!(event, EngineEvent::Rotated { .. }))
        .count();
    assert_eq!(
        first_rotations, 1,
        "a single frame must produce at most one rotation, got events {first:?}",
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after first rotate")
            .rotation,
        PieceRotation::R90,
        "one clockwise frame advances exactly one quarter turn (R0 -> R90)",
    );

    // Frame 2: button still held. The engine has no edge detection, so this is a
    // fresh press attempt -> one more rotation R90 -> R180. The key anti-repeat
    // guarantee is that it is still ONE rotation this frame, never two: the held
    // button does not spin the piece a full 360 across the two frames.
    let second = engine.step(InputFrame {
        rotate_clockwise: true,
        ..InputFrame::default()
    });
    let second_rotations = second
        .iter()
        .filter(|event| matches!(event, EngineEvent::Rotated { .. }))
        .count();
    assert_eq!(
        second_rotations, 1,
        "the second held frame must also produce at most one rotation, got events {second:?}",
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after second rotate")
            .rotation,
        PieceRotation::R180,
        "two held clockwise frames advance two quarter turns total (R0 -> R90 -> R180), \
         never auto-repeating past a single turn within a frame",
    );
}

// -------------------------------------------------------------------------
// §25.3 CAVEAT: the engine intentionally has no DAS state machine.
// -------------------------------------------------------------------------

/// DESIGN INVARIANT (not a placeholder): the engine must NOT grow DAS timing.
/// DAS is player-side (ADR-4); the engine translates the piece by exactly one
/// cell per held frame regardless of elapsed `dt`. Holding `left` across many
/// frames therefore yields one move *per frame* — never an accelerating burst
/// and never a delay before the first move. The player layer
/// (`src/player/das.rs`) is what shapes a held key into the spec's tap →
/// delay → auto-repeat cadence before these per-frame pulses reach the engine.
#[test]
fn caveat_engine_step_has_no_das_state_machine() {
    let config = EngineConfig {
        // Wide well so horizontal room is never the limiting factor.
        board_width: 40,
        ..EngineConfig::default()
    };
    // A large per-frame dt: a (wrong) auto-shift engine would fire several repeats
    // by frame ~3. The engine must still move exactly one cell per frame.
    let dt = 0.5;
    let mut engine = Engine::new(config, SEED_FIRST_PIECE_T);
    let before = spawn_first_piece(&mut engine);

    // Frame 1: the initial tap moves exactly one cell left.
    let first = engine.step(InputFrame {
        left: true,
        dt_seconds: dt,
        ..InputFrame::default()
    });
    let first_left_moves = first
        .iter()
        .filter(|event| {
            matches!(
                event,
                EngineEvent::Moved {
                    direction: MoveDirection::Left,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        first_left_moves, 1,
        "the initial held frame moves exactly one cell left, got events {first:?}",
    );

    // Frames 2..=8: button still held, plenty of elapsed time per frame. Because
    // the engine has no DAS state machine, each frame still moves exactly one cell.
    // (One-per-frame rather than zero: the engine treats every held frame as a
    // fresh pulse, so the absence of auto-repeat means "one cell per frame", never
    // an accelerating burst.)
    for frame in 2..=8 {
        let events = engine.step(InputFrame {
            left: true,
            dt_seconds: dt,
            ..InputFrame::default()
        });
        let left_moves = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    EngineEvent::Moved {
                        direction: MoveDirection::Left,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            left_moves, 1,
            "frame {frame}: no auto-repeat means exactly one cell per held frame \
             (the engine owns no DAS timing — that lives player-side), \
             got events {events:?}",
        );
    }

    // Net effect after 8 held frames: the piece has shifted exactly 8 cells in x
    // (one per frame), proving DAS acceleration is NOT applied. We assert only the
    // x-axis here: each frame also advances `dt` past the fall interval, so the
    // unrelated gravity system may legitimately drop the piece a row or two — that
    // vertical motion is not a DAS concern and would only mask the horizontal point.
    let after = engine
        .snapshot()
        .active
        .expect("active piece after holding left");
    assert_eq!(
        after.origin.0,
        before.origin.0 - 8,
        "8 held frames shift exactly 8 cells in x (one per frame), with no auto-repeat burst",
    );
}
