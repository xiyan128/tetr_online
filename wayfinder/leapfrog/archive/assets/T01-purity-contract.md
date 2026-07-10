# T01 — Purity contract (proposed, ratifiable)

The auditable checklist the final showdown bot must pass to count as "fully learned." Derived from the destination answer (fully learned deployed bot; no hand-tuned eval or search heuristics; CC2 only as round-0 warm start + yardstick) plus sensible defaults. **One genuine judgment call is flagged in bold** — everything else follows from the answer or a defensible default.

## The four categories

| Category | Rule | Rationale |
|---|---|---|
| **Environment truth** | ALLOWED | The game's own rules are not "tuning": attack tables, garbage/cancellation/cap, B2B/combo/PC scoring, the 7-bag law, seeded hole streams, topout detection, movegen (reachable-placement enumeration). |
| **Compute-budget knobs** | ALLOWED | Dials that trade wall-clock for quality without encoding *what is good*: beam width, MCTS `m`/`n`/sims, search depth, per-move wall-clock/node cap. Every learned planner (AlphaZero included) has a hand-set sim budget; the destination itself says "practical per-move budget." |
| **Training-time machinery** | ALLOWED | How you *train*, absent from the deployed bot: learning rate, replay mix, ε-exploration schedule, π-target temperature, Gumbel `g` noise, the completed-Q σ transform, seed regions, SPRT config. |
| **Hand-tuned eval / search heuristic** | FORBIDDEN in the deployed bot | Fixed numbers that encode *what is good* or *how to prune*: all CC2 eval terms/weights, `SPEC_DECAY`/`spec_weight` bag-optimism discount, any hand-set feature weighting, any hand-tuned move-ordering/pruning coefficient, the Z_SCALE/W/λ_att value-reward composition. |

## Component-by-component ruling

- **CC2 eval (`Cc2Evaluator`, `attack_tuned` weights)** — FORBIDDEN in the deployed bot. Permitted only as (a) round-0 warm-start/BC teacher and (b) the champion opponent/yardstick. ✓ direct from the answer.
- **`SPEC_DECAY = 0.75` / `spec_weight`** — FORBIDDEN (hand-tuned search heuristic). Replaced by the learned value's own bag-belief / the survival-CVaR chance backup (T02). The exact-enumerated chance expectation is *environment truth*, not tuning.
- **Z_SCALE / W / λ_att value-reward composition** — ELIMINATED by design, not merely allowed: the terminal-WDL value target (T02) makes attack's value *learned* (instrumental to winning), so there is no hand-set composition in the deployed bot. Any transitional composition is a training-only crutch that must be gone by the showdown.
- **`DEATH_SCORE` sentinel** — ALLOWED. It is not an eval weight; it encodes the environment truth "a topped-out line is terminal and maximally bad," which the terminal-WDL target already says. It is a search-internal representation of `z = −1`, not a tuned number.
- **Beam width / MCTS `m`,`n` / depth / wall-clock** — **ALLOWED as budget knobs (the one genuine judgment call; default = allowed).** *If the user instead rules a hand-chosen width a forbidden fixed component, the search itself must learn to allocate its own compute — a materially harder system. The default (budget knobs allowed) matches AlphaZero practice and the "practical per-move budget" destination framing; flagged here for veto.*
- **CVaR risk level `α`** — ALLOWED but constrained: it is a single interpretable risk dial (like a budget knob), NOT free eval tuning — but it must be **ablated and justified, never fit to the benchmark** (the α→mean ablation is pre-registered in T02). Ideally it becomes learned/annealed; a fixed sensible α in the deployed bot is acceptable if ablated.
- **Venue constants in obs (features 83..85: rain clock, cap fraction)** — ALLOWED. These are the *environment definition* the net observes, not tuning. Consequence (not a violation): a trained net is bound to the calibrated venue; re-calibrating the venue mid-campaign invalidates the net. Noted for the gate battery (T04).
- **Movegen, hold, from_snapshot** — ALLOWED (environment mechanics).

## The showdown audit (what "fully learned" is checked against)

The deployed bot at the final showdown must contain, in its decision path, **zero numbers from the FORBIDDEN row** — verifiable by grep/inspection of the deployed arm: no `Cc2Evaluator`, no `SPEC_DECAY`, no composition weights; only (learned net weights) + (budget knobs) + (environment rules). CC2 may appear only as the *opponent* in the race harness, never in the bot.

**Status:** proposed default contract. Ratify or veto the flagged budget-knob call; otherwise this is the T01 answer and the design freeze (T08) proceeds on it.
