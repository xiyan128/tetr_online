# ADR: The value-net strike

Status: **in progress** (worktree `valuenet/strike`). A deliberate moonshot — the cheap E8
gate leaned STOP (no compelling headroom on the survival proxy), and we are opening the
expensive front anyway, on the bet that the headroom the aggregate gate dilutes lives in the
rare hard positions a learned, raw-input eval could price.

## Goal

A learned leaf evaluator distilled from the **best expert we have**, taking **all raw board
information** as input, replacing the handcrafted CC2 `Value` at the beam's leaves. The bar is
the **iron gate** (E9): match CC2 at iso-search on held-out seeds *before* any depth is added —
because deep search amplifies a bad eval as readily as a good one.

## What killed the last attempt (and how we avoid it)

The pruned `tetr-nn` stack (Burn MLP `8→64→64→1`, JAX trainer; `git show a5820b5`) failed for two
reasons named in `docs/value-net-postmortem.md`:

1. **Weak teacher.** It distilled DT-20 (~0.2 APP). A net can't exceed its teacher's ceiling.
   → **Fix:** distill the **champion's own deep search** (`tp128d9`, ~0.82 APP) — the search-backed
   value `root_best`, not a weak handcrafted scalar. Optionally a stronger oracle (`w256d12`).
2. **No death coverage.** Distilled from a survivor's games → almost never sees dying boards →
   never learns "danger ⇒ low value" → stacks into top-out.
   → **Fix:** generate trajectories **under rain** (the survival pressure that manufactures
   near-death states), reusing the `rain_state_bank` insight. Over-sample danger.

Two more constraints it implies:
3. **Determinism + wasm.** Inference must be **fixed-point `i32`** in the `SCALE=256` domain
   (bit-identical across opt levels / native / wasm), **no Burn at inference** (the ~4k-line dep
   tree was the real prune reason). Training framework is free; inference is hand-rolled integer.
4. **All raw input.** The `Evaluator` trait sees only `(lock, board, t_spin, ctx{combo,b2b})` —
   *not* queue/hold/bag/pending. A raw-input net needs the full `SearchState`, available at
   `score_child` (`search/mod.rs:154`). Integration extends the seam (a `StateEvaluator` that
   reads the whole state), not the column-only trait.

## Architecture (clean, separate subsystem)

```
crates/tetr-valuenet/            # the Rust subsystem (native research + future wasm inference)
  src/encode.rs                  # THE input schema: SearchState -> tensors. Single source of
                                 #   truth, shared by export AND inference (no drift).
  src/sample.rs                  # one training record: features + (value, policy, outcome) labels
  src/dataset.rs                 # safetensors shard writer/reader (the Rust<->Python interchange)
  src/infer.rs        (M3)       # fixed-point i32 forward pass + the StateEvaluator seam
  src/lib.rs
python/valuenet/                 # training (uv + PyTorch); reads shards, writes weights
  pyproject.toml                 # uv-managed; torch, safetensors, numpy
  data.py  model.py  train.py
crates/tetr-research/...         # `bc-distill` command: drive the teacher under rain, encode,
                                 #   label with the teacher's search value+move, write shards
```

**Tech stack (best + clean):** PyTorch for training (most mature/maintainable for a board CNN),
`uv` for the env, **`safetensors`** as the single typed interchange both languages speak (shards
in, weights out), hand-rolled fixed-point `i32` Rust inference for determinism + wasm.

## The input schema (all raw information)

Per position (the board *after* a placement — the leaf the eval prices):
- **Board planes** `[C,H,W]`, `H=24` (20 visible + 4 buffer), `W=10`, from `board.columns()`:
  plane 0 = occupancy. (plane 1 = garbage cells, from `EngineSnapshot.board_cells.garbage`, if it
  pays — deferred; occupancy first.)
- **Piece categoricals** (7-way each): active type, hold (+empty), queue[0..5], bag remainder
  (7-bit multi-hot).
- **Chain scalars:** combo, b2b, pending-garbage total lines + per-column incoming-hole indicator.

`encode.rs` is the one place this is defined; export and inference both call it.

## Targets (distill the best expert)

Driven by the teacher under rain; at each piece run `think_to_completion(teacher, state, budget)`
→ the move IS the trajectory step AND the label:
- **value** = `plan.score` (the teacher's deep-search valuation, in `SCALE` units) — the primary
  leaf-eval target (E9).
- **policy** = `plan.placement` (the teacher's chosen move) — for a later move-ordering prior (§4.2).
- **outcome** = realized future attack + `died_soon` (window survival) — for a later win/outcome
  value (the two-agent-aware target the survival gate hinted CC2 already half-captures).

## Milestones (each gate-green, committed in the worktree)

- **M0 — foundation:** the `tetr-valuenet` crate, `encode.rs` schema + tests, `dataset.rs`
  (safetensors), the `bc-distill` exporter. *Produces a real dataset.*
- **M1 — train + native float infer:** PyTorch board-CNN value net; held-out value-MSE; a Rust
  f32 inference + `StateEvaluator` seam; **race vs CC2 at iso-search (the iron gate)**. Float is
  fine here (native research); proves headroom-or-not.
- **M2 — fixed-point + quantize:** `i32` forward pass, bit-identical; re-confirm the gate; wasm
  size check.
- **M3 — autoresearch:** `/autoresearch` over architecture / targets / death-coverage / depth.
- **M4 — close the loop (stretch):** outcome/win value + self-play to exceed the teacher.

Honest odds (roadmap §4.1): ~40% it matches CC2 in native research, <20% it ships in-game. The
gates (iso-search match before depth) are designed to kill it cheaply if it can't clear them.
