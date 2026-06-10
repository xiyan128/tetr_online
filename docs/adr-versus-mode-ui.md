# ADR: Versus mode in the game — seats, real-time exchange, and the two-board UI

Date: 2026-06-10 · Status: accepted (implemented on `feat/versus-mode`)

## Context

The engine already owns every versus *rule* (`docs/adr-versus-rules.md`): the
pending queue, cancellation, rising, hole streams, and the `AttackSent` /
`GarbageInserted` events. The search models the full garbage transition. What
does not exist is a **playable surface**: the game is single-board end to end —
one `EngineState` resource, one camera centered on one playfield, reconcilers
that despawn render blocks by global marker, one keyboard latch, one score HUD.

This ADR covers the game-side design: how two simultaneous engines live in the
Bevy world, how attack routes between them in real time, who can sit at a
board, and what the player sees. v1 ships **human vs bot** and **bot vs bot**;
the seat interface must already be shaped so human vs human (local split
keyboard, then remote) drops in without re-architecting.

## Decision 1: versus is a sibling state, not a flag on `Playing`

`GameState` gains `VersusSetup` (the pre-match screen) and `Versus` (the
match). `Versus` carries its own sub-state machine:

```
VersusPhase: Countdown → Running ⇄ Paused, Running → Over
```

The single-player `level` module is **not touched**: its setup, drivers and
reconcilers are all scoped to `Playing`/`PauseState`, which never fire in
`Versus`. Versus gets its own module (`src/versus/`) with multi-board-native
systems. The duplication this buys is small (the reconcilers are ~40 lines
each) and the risk it retires is large: the polished single-player path keeps
its byte-identical pipeline and its deterministic-schedule tests.

The recorded follow-up (not v1): once versus stabilizes, single-player can be
re-homed as a one-seat match on the same architecture, retiring the global
resources. The versus systems are written generically over seat entities
precisely so that move is mechanical.

`VersusPhase::Over` stays **inside** `Versus` — the final boards remain on
screen under the result banner (reading the losing stack is half the fun), and
a rematch rebuilds the session in place instead of bouncing through a state
transition that would despawn everything first.

## Decision 2: seats are entities; the participant is an enum (open set)

A match is two **seat entities**, each carrying its own engine and its render
anchor:

```rust
Seat { index }                 // 0 = left board, 1 = right board
SeatEngine(Engine)             // the authoritative simulation, per seat
SeatSnapshot(EngineSnapshot)   // published after every step
SeatEvents(Vec<EngineEvent>)   // this frame's events, per seat
SeatStats { attack_sent, .. }  // cumulative, for the HUD and result screen
```

Who controls a seat is configuration, not architecture:

```rust
enum Participant {
    Human,                  // the local keyboard (one human seat in v1)
    Bot { model: usize },   // an index into the existing ModelRegistry
    // future: RemoteHuman { .. } — frames arrive from a net channel
}
```

`VersusConfig { seats: [Participant; 2], seed: Option<u64> }` is written by the
setup screen and read once when the match spawns. Everything downstream
(stepping, rendering, HUD labels) reads seat components, never the config — a
future participant kind only has to produce an `InputFrame` per fixed slice.

Bot controllers are `Send`-but-not-`Sync` (the established `AiPlayer`
precedent), so they live in one non-send resource, `VersusBots`, keyed by seat
index — not as components.

**Both engines get the same piece seed.** Identical bags are the guideline
fairness convention: the match measures placement skill, not draw luck. The
hole streams stay decorrelated per receiver by the engine's own salt. The seed
is fresh per match (entropy from app-clock nanos at spawn; a test override sits
in `VersusConfig.seed`), and a rematch draws a new one — replaying the exact
same deal is a replay feature, not a rematch.

## Decision 3: real-time exchange — step both, then route, symmetrically

Each `FixedUpdate` slice (the same 60 Hz clock as single-player):

1. **Step both engines** with their participant's frame (human: the latched
   keyboard exactly as `step_engine` does it; bot: `controller.poll()`).
2. **Route attack**: every `AttackSent { lines }` from seat A is
   `queue_garbage(lines)` on seat B, both directions, after both engines have
   stepped.
3. **Publish snapshots** (so meters and boards show the routed garbage the
   same frame) and accumulate events for the reconcilers.
4. **Check death**: a seat whose snapshot reports `game_over` loses; both in
   the same slice is a draw. Attack is routed *before* death is read — the
   engine already guarantees a dying lock sends nothing, and the driver never
   second-guesses events (the `play_versus` ruling).

Step-then-route means an attack lands one slice (16.7 ms) after the clear that
sent it, identically in both directions — neither seat order nor system order
can advantage a side. The engine ignores `queue_garbage` once dead, so routing
into a just-dead seat is safely inert.

**Known asymmetry, accepted and bounded:** the bot controller emits `dt = 0`
maneuver frames (positioning advances no gravity — the established Watch-AI
convention that keeps plan execution exact). In a wall-clock match the bot's
engine therefore experiences slightly stretched time (~15–20% during its short
maneuver bursts). Versus pins **flat level-1 gravity** (Decision 4), where the
dilation is imperceptible — and the alternative (stamping real `dt` onto
maneuver frames) would desync rendered plans from the board, a correctness
class of bugs, traded for fairness nobody can see. Revisit only if a
high-gravity versus variant ships; the honest fix then is replanning under
gravity, not dt-stamping.

**The bot plays blind to the queue, deliberately.** The snapshot handed to a
bot has `pending_garbage` cleared. This is not a shortcut: the experimental
record (memory + `versus_climb` header) shows the garbage-aware search is
*decisively worse* under pressure with today's no-garbage-world weights — the
model is right, the prices are wrong. Blind is currently both the stronger
*and* the fairer opponent (it cannot exploit perfect hole information against
a human, who only sees the meter). The engine still cancels and rises by rule
regardless. When re-priced weights land, awareness becomes a per-model flag in
the registry entry, not a redesign. (Core still gains the
`PieceSignature` pending term now, so an aware bot replans correctly the day
it ships.)

## Decision 4: versus rules of play

- **Flat gravity**: `GoalSystem::None` (new engine variant — no goal, no
  leveling), starting level 1. Guideline versus does not speed up; pressure
  comes from the opponent, not the clock.
- **Garbage cap**: the engine default (8 per clear-less lock).
- **No variant end conditions**: the match ends when a seat dies. No score, no
  high-score entry — the outcome *is* the score.
- **Hold/preview/lock-down**: the player's existing `GameSettings` apply to
  both seats (symmetric rules; the bot uses the same preview depth it would in
  Watch-AI).
- **Pause** pauses the whole match (both boards freeze, overlay offers
  Resume/Quit). In a local match pausing is inherently mutual; a remote future
  replaces pause with forfeit, which is why pause lives in `VersusPhase`, not
  in a seat.

## Decision 5: the two-board UI

Layout (board coordinates, one `BoardRoot` entity per seat; everything a seat
renders is a **child** of its root, so position is one transform and despawn
is one subtree):

```
   [YOU]                            [BEAM CC2]
HOLD ┃##########┃ NEXT     HOLD ┃##########┃ NEXT
 □□  ┃          ┃  ▢        □□  ┃          ┃  ▢
     ┃          ┃  ▢            ┃   ...    ┃  ▢
   ▌ ┃   ...    ┃  ▢          ▌ ┃          ┃  ▢
   ▌ ┃          ┃             █ ┃          ┃
ATK 12┗━━━━━━━━━━┛         ATK 34┗━━━━━━━━━━┛
          3 … 2 … 1 … GO! / YOU WIN!
```

- **Two identical board groups** (hold column · 10×20 field · preview column),
  seat 0 left, seat 1 right, separated by a 6-cell gutter. Identical
  orientation v1 (mirroring is a per-root parameter later, not a layout
  rewrite).
- **Camera**: one `Camera2d` with `ScalingMode::AutoMin` sized to the full
  scene, so both boards always fit — native window resizes and the web canvas
  get the same framing for free. Tagged `GameplayCamera` so the CRT/bloom
  passes apply unchanged.
- **Garbage meter** — the signature versus readout: a thin vertical bar on
  each board's inner edge, one red segment per pending line, stacked
  bottom-up with a 2-px notch between batches (you can read "a 4 and a 2 are
  coming" at a glance, like the guideline games). Reconciled from
  `snapshot.pending_garbage` every frame; it empties on cancellation and
  drains into the board on rise.
- **Garbage rows render gray.** Garbage gets an honest `CellKind::Garbage` in
  the engine and a `garbage` flag on `SnapshotCell`; the renderer paints those
  cells neutral gray. Reading your stack means instantly separating your
  pieces from their attack; cyan garbage (the current `GARBAGE_FILL`) is
  illegible.
- **Seat labels** above each board: "YOU" for the human seat, the registry's
  short model label for bots.
- **ATK counter** under each board: cumulative attack sent (the per-seat
  pressure scoreboard; APM needs a timer baseline a 60 Hz HUD can derive
  later).
- **Countdown**: 3 · 2 · 1 · GO! center-screen (0.8 s per beat); engines hold
  (no spawn) until GO, so both first pieces appear simultaneously.
- **Result banner**: dim scrim over the *live final boards* — "YOU WIN / YOU
  LOSE / <MODEL> WINS / DRAW", per-seat attack totals and match time, and
  `Enter` rematch / `Esc` menu. No high-score flow.
- **Sound design**: move/rotate/drop SFX play for the **human seat only** (a
  bot's input stream is noise, and bot-vs-bot would be a drum roll); line
  clears play for both seats; a garbage **rise** gets the lock-thunk cue — the
  thing you must hear is your own board getting heavier. Attack-sent shows a
  brief "+n" pop by the sender's board.

### Setup screen (`VersusSetup`)

The existing `FocusList` menu idiom, four rows:

```
        VERSUS
  P1   ‹ You ›
  P2   ‹ Beam DT-20 ›
  Start
  (Esc back · ←/→ change · Enter start)
```

`←`/`→` on a seat row cycles its participant (P1: You + every registry model;
P2: registry models — exactly one human seat in v1 because there is one
keyboard). Defaults: You vs Beam DT-20 (a mid-strength opener; the picker
remembers the last choice for rematch parity). Choosing a model for P1 gives
bot-vs-bot — the versus twin of Watch-AI.

## Consequences

- Single-player behavior is bit-identical (its module is untouched; the new
  engine `GoalSystem::None` and `CellKind::Garbage` paths are unreachable from
  it except the inert `garbage: false` snapshot flag).
- The research crate keeps `BlindToGarbage` and `play_versus` as instruments;
  the game's blinding is a one-line snapshot strip in its own bot driver
  (no game→research dependency).
- Two engines step per slice: ~2× engine cost (µs-scale) and the bot search is
  already sliced (`SlicedRunner`, 16-node quanta) — versus stays inside the
  frame budget the anytime-search ADR established. Bot-vs-bot runs two sliced
  searches per frame; the budget tests cover the sum.
- The seat architecture is the declared landing zone for human-vs-human:
  local = a second keymap producing a second seat's `InputFrame`s; remote = a
  participant whose frames arrive from a channel plus a forfeit rule. Neither
  changes the match core.

## Deliberately deferred

- **Messiness / hole-change models** — tuning, after re-priced weights.
- **Aware bots in the registry** — blocked on re-pricing (the mispricing
  finding); the flag-and-signature plumbing ships now.
- **Best-of-N match format** — the `SeatStats`/outcome plumbing supports it;
  v1 is single game + rematch.
- **Mirrored seat-1 layout**, replays, spectator APM panels.
- **TBP referee alignment** for external bots in-game.
