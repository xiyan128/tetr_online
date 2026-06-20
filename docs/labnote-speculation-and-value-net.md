# Lab note — speculation backup (E3 / E11) + value-net Phase-0 (E8): the cheap in-paradigm levers are exhausted

*2026-06-20. Built, tested, and committed (`d382378`) on `perf/ai-search-perf`, then reverted —
decision: do **not** integrate the code; this note is the durable record. The implementation is
recoverable from git reflog (`d382378`).*

## Context

The §2 roadmap localized the ~1411-Elo ceiling and named two in-paradigm escapes: **better
speculation** (belief over the unseen 7-bag) and a **better whole-eval** (a learned value net).
Both were probed with cheap gates this session. Both came up empty.

## E3 — `SPEC_DECAY` value sweep: the discount scalar is exhausted

Swept the per-ply speculative reward discount `{0, 0.5, 0.75, 0.9, 1.0}` at w16d12 vs the `0.75`
baseline (rain pair-GSPRT). All **inconclusive** — no value beats `0.75`, and *full optimism (1.0)
is the worst* (LLR −2.36 toward the baseline). The discount magnitude is not the lever: the
*undiscounted board Value* carries speculation (zeroing the reward, `dec0`, is ~flat). This points
at the backup **structure** — the `max` over the bag draw — not the scalar. → E11.

## E11 — expectimax backup: the principled fix is a wash

Replaced the flat optimistic-`max`-over-the-bag with an **expectimax (bag-mean) backup** at the
empty-queue boundary — rigorous (chance = full bag-uniform mean, MAX width-truncated, board Value
kept whole; unit-pinned by depth-1 exact equality with the flat backup, determinism, and the
`E[max] ≤ max[max]` inequality). **Result: NULL.**

- vs the flat d12 baseline it loses (H0, LLR −3.0…−3.5) — but mostly on **depth** (expectimax's
  full, un-truncated chance fan-out caps its tail at 2–3 plies vs flat's ~6).
- At **matched** speculative depth it's a **wash** (`exm-w16d2 ≈ flat-d7`, +1.86; `exm-w2d3 ≈ flat-d9`).
- **Due diligence** (the null survived a bug-hunt): an off-by-one depth match was found and fixed
  (the boundary sits at depth 5, so flat `dN` does `N−5` draws); width starvation ruled out
  (exhaustive `usize::MAX` ≈ width-2, both 92% ply-1 decision agreement with flat); death-poisoning
  ruled out **even on near-death rain boards** (0/60 death-dominated).
- **Mechanism:** the optimism bias is *near-uniform across the ply-1 candidates*, so max-vs-mean
  shifts absolute values but not the **decision** (the choice is dominated by the 6 shared concrete
  plies). The "Hope Tetris" framing was a **misdiagnosis** — the cheap `max` is a good
  approximation: equal in belief at matched depth, and cheaper, so it reaches deeper.

## E8 — value-net Phase-0 gate: leans STOP

A registered eval (`value-probe` / `value-probe-heavy`) + a `Cc2Evaluator::board_value_scaled`
accessor: play the champion under rain and regress realized **attack** *and* **death-in-window** on
the static CC2 Value, inline (no training set).

- The **attack** proxy looked like headroom (Value R²=0.003 vs board-height R²=0.17) — but that is a
  **downstacking confound** (a taller board has more garbage to clear → more attack); a net trained
  on it would value tall boards, which is suicidal.
- The confound-free **survival** gate flips it: on death-in-window, CC2 Value **beats** the best
  single board feature (death~Value R²=0.26 > height 0.22, champion). **CC2 already prices survival.**
- So the cheap gate shows **no compelling value-net headroom on the real objective.** E9 (the
  multi-week distill + iso-search match) is *not* justified by Phase-0.

## Verdict

Both named in-paradigm escapes are exhausted at the cheap-gate level — the speculation back-up
(neither the decay scalar nor the max→mean structure moves survival) and the whole-eval (CC2 already
prices survival). Together with the later **adaptive-allocation null** (depth saturates ~d12, so
there is nothing to reallocate into — see [labnote-adaptive-allocation.md](labnote-adaptive-allocation.md)),
the bot is **near its ceiling for cheap moves**. The remaining frontier — a learned eval (this gate
leans STOP), adaptive *width*, two-agent / timing search — is a deliberate, expensive commitment the
evidence does not green-light. The right next step is a *bet*, not a cheap gate.
