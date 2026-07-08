# T08 — Leapfrog system design (PROPOSED freeze, awaiting ratification)

Synthesis of the resolved tickets: algorithm ([T02](T02-algorithm-family-memo.md)), purity ([T01](T01-purity-contract.md)), throughput ([T03](T03-throughput-measurements.md)), and the two green preconditions (Gate-0a coverage 3.1×; Gate-0b search-gain 31-1 on clean seeds). This is the leapfrog's STAGE1-DESIGN equivalent. **Marked PROPOSED, not frozen — it is a concrete artifact to ratify or amend, with genuine user choices flagged ⚑.**

## 1. The learning system

- **Family:** Expert iteration with a policy-guided search over the *true* tetr-core simulator (not MuZero — we own the exact model). Value + policy net; search improves the policy; train on search outputs; repeat.
- **Search vehicle (deployed + datagen):** policy-guided search selecting by **completed-Q**. v0 = the existing beam restricted to the net's top-m policy roots (correct max-backup Q via `root_scores`, reuses tested code). The Sequential-Halving efficient MCTS is a **deferred throughput refinement** (T12), not required for soundness — the completed-Q *target transform* is the actual fix for the near-deterministic-backup trap, and it is Python-side over stored root scores.
- **Value target:** terminal-WDL `z ∈ {+1,0,−1}` from topout / sudden-death only — unhackable (kills APP-farming and attack-suicide). Eliminates the Z_SCALE/W/λ composition entirely (purity ✓).
- **Policy target:** `π' = softmax(logit + σ(completedQ))` over all roots, completedQ from the search's per-root Q. This is the panel's dodge of fact (d): smooth even when the argmax is near-deterministic.
- **Chance handling:** exact-enumerated bag/garbage expectation with a **survival-CVaR_α** backup replacing the forbidden SPEC_DECAY (purity ✓); `α→mean` is the pre-registered ablation. (v0 may use the beam's existing speculation; CVaR lands with the Gumbel MCTS refinement.)
- **Opponent representation:** factored `V_versus ≈ V_solo_survival + λ·A_timing(compressed opp summary)`; the search stays single-agent over my board (justified: ≤6% mirror coupling without rain). The **R4 fix** is mandatory: the deployed net bot's vehicle must consume the policy prior (retire value-only `compose()`).
- **Net:** two-headed (value + ~34-way policy) + an SSL board-reconstruction aux head (lifts value effective-N games→states, attacks the round0 WDL AUC 0.607). Obs = the existing two-board encoder.
- **Warm start:** round-0 = BC of value+policy from CC2/champion rollouts (the only CC2 use); round 1+ targets self-generated.
- **Self-play pool:** 70% latest / 20% prioritized-past / 10% **champion-pinned** (aligns datagen with the graded race); prioritized replay keyed on KL(π'‖π_θ). No full league unless a 1-D-Elo antisymmetric-residual tripwire (>10%) fires.

## 2. Throughput plan (from T03)

- **Fixing the glue-bound forward (T13) is a prerequisite** — ~6.9k evals/s today, ~99% im2col/transpose glue. Target ≥200 games/hr for a strong-ish agent before authorizing a campaign.
- Keep per-move eval count low (top-m + eventual Sequential Halving). Deep-wide net beams are datagen-infeasible locally.
- Actor/learner split, containerized worker = the same actor at n=1 locally and n=many on cloud; per-game shard I/O. ⚑ **Cloud provider still unpicked** (destination said provider-agnostic) — a decision when a round outgrows the Mac.

## 3. Gate battery (proposed defaults — folds in T04) ⚑

- **Versus claim:** pre-registered pair-GSPRT vs `probe-tp128d9` under the sudden-death venue (rain 8, cap 240), at **matched ~100 ms/move wall-clock**. ⚑ **On what hardware, and is ANE-vs-CPU fair?** (Precedent accepted ANE for the decisive conv_rb1 race — proposed default: ANE allowed for the net, matched wall-clock, champion at its best with the spec-dedup applied.)
- **Solo claim:** marathon-holdout APP (champion 0.8225) + downstack, censored metrics; solo APP never a *sole* verdict (combo-farmable).
- **Statistics:** the existing latched trinomial gate (p1=0.55, α=β=0.05); ⚑ **is p1=0.55 sensitive enough for the final claim, or tighten** given leaf-edge compression made large edges read EVEN?

## 4. Rigor / cadence (proposed defaults — folds in T06) ⚑

- **Frozen:** the purity contract, seed-region discipline, the venue, the gate battery (once ratified).
- **Floating:** net architecture, temperatures, schedules, α.
- **Halt philosophy:** keep A12 (halts = breakage tolerances, wider than calibration targets).
- **Cadence:** ⚑ **the prior campaign's heavy pre-registration caught 7 bugs but shipped 0 rounds in 14 days.** Proposed default: lighter contract — pre-register the gate + kill criteria (already done), but let architecture/schedules float with a one-line amendment log, not a frozen contract per deviation. One resumable command per round (port the `rounds.py` discipline). A round *aborts* (not slips) if first-hour datagen < 150 games/hr (throughput STOP) or start-gates fail.

## 5. Pre-registered kill criteria (from T02, still binding)

Target-extraction STOP (completed-Q targets regress the net → calibration wall); Deployment-parity STOP (π-guided search loses at matched 100 ms → relax to "beat at a larger budget" or abandon); Throughput STOP; Value-collapse STOP (std(z_hat) < 0.15).

## 6. The genuine decisions awaiting the user ⚑

1. **Purity budget-knob veto** (T01) — is hand-chosen search width/sims an allowed budget knob (default yes) or a forbidden fixed component?
2. **Gate hardware + ANE fairness** (§3) — what hardware defines "matched ~100 ms/move," and is ANE-for-net-vs-CPU-for-champion fair?
3. **Rigor/velocity trade** (§4) — the lighter-contract default vs the prior heavyweight pre-registration.
4. **Gate sensitivity** (§3) — keep p1=0.55 or tighten.

Everything else is proposed to proceed on the defaults above. On ratification, the campaign-build tickets (datagen driver, completed-Q training, round driver) get charted and the strike moves from planning to execution.

---

## RATIFIED 2026-07-08 (user delegated all four; explicit steer on rigor/velocity)

1. **Purity budget-knob:** ALLOWED (search width/sims/wall-clock are compute knobs).
2. **Gate:** same machine, matched ~100 ms/move, each bot its best inference path (champion CPU beam + spec-dedup applied; net ANE/BLAS); hardware reported in the receipt.
3. **Rigor/velocity — BALANCED, INFRA-FIRST (user):** "optimize for breakthrough but invest in ways to make research faster — potentially optimize/refactor infra before jumping to a task." Contract: freeze the gate + kill criteria only; architecture floats with a one-line amendment log. **Research-velocity infra (throughput, datagen/experiment tooling) is first-class work, done before grinding rounds — not deferred.** The first execution step is therefore an infra investment (the throughput fix), not a training round.
4. **Gate sensitivity:** p1=0.55 latched for round-to-round promotion (fast); the final showdown vs the champion uses a larger pair budget and reports effect size + CI, not just a binary verdict.

The design is FROZEN on these terms. The strike moves to execution, infra-first.
