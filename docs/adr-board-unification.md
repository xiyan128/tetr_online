# ADR: one board — BitBoard occupancy + a colour plane

Date: 2026-06-10 · Status: accepted (implemented on `refactor/board-unification`)

## Context

The engine carried two complete board implementations: `Board`
(`Array2D<Cell>`, where every `Cell` redundantly stored its own `(x, y)`
beside its `CellKind`) and `BitBoard` (column bitmasks, the search's `Copy`
fork currency). Line clearing, garbage insertion, full-row detection, and
skyline queries were each implemented twice, kept aligned by five randomized
differential tests — a standing two-copies-must-agree liability the audit
flagged as the engine's biggest structural debt. Every consumer was already
representation-split: collision/movegen go through the `Occupancy` trait,
evaluators read column bits, and only the snapshot needs per-cell colour and
garbage identity.

## Decision

`Board` becomes a thin composite: a `BitBoard` for **occupancy** (the single
home of every rule: full rows, compaction, garbage insertion, overflow,
skyline) plus a flat **colour plane** (`Vec<CellKind>`, row-major over the
backing grid) for per-cell identity. Mutations update both in lockstep;
reads dispatch to whichever plane answers (occupancy questions to the bits,
identity questions to the colours). The rule implementations that lived on
the dense board are deleted — `Board::clear_lines` and
`Board::insert_garbage_lines` now *drive the colour plane from the
bitboard's own results* (its cleared-row list, its overflow verdict), so the
rules cannot disagree with the search's view by construction.

The `Cell` struct (24 bytes for 1 byte of information) is deleted along with
its kind-ignoring `Eq`/`Ord` impls; cell iteration yields computed
`(x, y, CellKind)` tuples.

Envelope: the bitboard's `width ≤ 16, backing rows ≤ 64` becomes the board's
envelope, asserted at construction with a clear message. Every constructor
in the workspace fits (max in use: 10 × 40).

## Consequences

- The five cross-representation differential tests retire; in their place,
  one **internal coherence property** (occupancy derived from the colour
  plane equals the bitboard, under randomized op sequences) guards the
  lockstep invariant.
- Engine stepping reads occupancy through bits (collision, T-spin corners,
  row counts) — faster, though the engine was never the hot path; the
  search's `BitBoard` usage is unchanged.
- `Board` clones drop from ~9.6 KB memcpy to ~128 B bits + a 400 B colour
  vec (only the engine clones boards; the search forks bare `BitBoard`s).
- The acceptance/guideline suites run unchanged — the `Board` API surface
  (`set`, `get_cell_kind`, `cells`, `column_bits`, `lock_and_clear`, …) is
  preserved over the new internals, and the bin goldens pin behaviour
  end-to-end.

## Rejected

A pure-bitboard board with colour reconstructed at snapshot time — colours
are not derivable from occupancy; the snapshot's per-cell identity (piece
colours, the garbage flag) is real state and needs a real plane.
