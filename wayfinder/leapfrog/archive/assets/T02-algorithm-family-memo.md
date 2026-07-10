# T02 — Algorithm-family design memo

**Question:** which learning-system family fits fully-learned SOTA Tetris-versus from first principles?

**Method:** six independent expert lenses (ExIt/AlphaZero, Gumbel, stochastic-planning/chance-nodes, league/multi-agent, sample-efficiency, first-principles skeptic) → adversarial red-team cross-examination → synthesis. All lenses grounded against the repo's hard evidence; the red-team traced and refuted two over-claims (a "fresh" G_π=0.900 that is the same contaminated 99-11-0; a 339-393 games/hr figure the campaign's own R6 puts at 150-250).

---

## Recommendation: **Gumbel Expectimax-AlphaZero on the true model** — conditionally authorized, gated behind a cheap zero-training falsification.

A policy-**guided** search over tetr-core's exact simulator, with four load-bearing design commitments and one non-negotiable structural change:

1. **Gumbel-top-m root sampling + Sequential Halving with completed-Q — NOT PUCT/visit counts.** This is the crux that dodges the repo's fact (d) (backups near-deterministic at width). Grill 2020: the PUCT visit distribution ≈ argmax over `Qᵀy − λ_N·KL(y‖π_θ)`, and as backups sharpen `λ_N → 0`, so the visit-count policy target collapses to the beam's near-one-hot argmax — re-deriving exactly the target that already starved. Gumbel's **completed-Q** target reads Q over *all* ~34 placements, so the improved policy `π' = softmax(logit + σ(completedQ))` is non-degenerate even at 90%+ visit concentration. This is the precise mechanism the prior beam-root approach lacked.

2. **Exact enumerated chance nodes with a survival-CVaR backup — deletes SPEC_DECAY=0.75.** The beam's `expand_speculative` maxes over the bag draw — a category error (you choose your placement, you do *not* choose the bag). We own the exact simulator and the chance distributions are closed-form (uniform-without-replacement over the ≤7-piece bag remainder; the salted-PRNG hole column), so we compute the chance expectation directly. Replace mean-expectation with a **lower-quantile / CVaR_α (α≈0.25)** over outcomes: under sudden-death rain the object that matters is P(survive the worst draws), and optimistic-max is the exact wrong tail. This is the only concrete answer to the skeptic's decisive objection ("you cannot distill a hedge into a point estimate") — the tail-coverage lives in the **value** (a bag-conditioned lower quantile), so a *narrow* search need not span the bag by breadth. `α→mean` is the pre-registered null/ablation.

3. **Terminal-WDL value target — unhackable.** `z ∈ {+1,0,−1}` from topout / sudden-death resolution only. This structurally kills fact (f): APP combo-farming and attack-suicide-under-rain both vanish because the reward literally *is* surviving-and-winning, not attack or clears. Negamax through decision nodes, CVaR-expectation through chance nodes; optional TD(λ)/search-value bootstrap with a Monte-Carlo-z anchor early. Add a per-**state** SSL board-reconstruction aux head to lift value's effective-N from games (~2k) toward states (~600k), attacking the round0 net's measured WDL AUC 0.607.

4. **Factored two-board value, compressed opponent.** `V_versus ≈ V_solo_survival(my board, my pending) + λ·A_timing(compressed opp summary)` — never the full opponent board. Justified by the repo fact that mirror coupling is ≤6% without rain (weak, delayed interaction); negamax on the joint state under the venue's alternating-first-mover approximation.

**The non-negotiable structural change (R4 fix):** the deployed net bot's search vehicle **must consume the policy prior**. Today `compose()` reads only `net_value`, so *any* policy-head signal certifies a head the deployed value-driven beam never consults. The value-only TP-beam is retired for the net bot; the vehicle becomes this π-consuming search. Without this, the entire policy axis is unobservable in deployment.

---

## Why it *might* leapfrog — stated honestly

The design **refuses to claim** it beats the champion at matched wall-clock. Facts (a) width-as-survival-hedge and (b) leaf-compression-to-even were both measured **at w128d9**, where a bounded search has already resolved everything inside the horizon. The leapfrog **bet** is that at *shallow width with a strong survival-relevant prior*, the survival Q-spread (a hole-making placement vs a clean one within 2–3 plies) is **not yet compressed** — i.e. w128 flatness is a depth artifact. If a learned prior concentrates the hedge into m=8–16 candidates, Sequential Halving spends the whole budget adjudicating exactly those, and shallow-guided m=8 makes brute w128 **unnecessary** rather than out-muscling it.

This is unmeasured, and the existing low-width evidence points the *other* way — but those narrowings (w6-pruned lost 2-9; narrow-deep dies 18-22%) were by **myopic value**, not by **learned survival-relevance**. That distinction is the exact untested door.

## The gate that opens or closes the door — before any campaign spend

**GATE-0 (zero training, ~1 day) is the next action, and it precedes even the round-0 retrain:**

- **0a — survival-recall@k:** on *existing* champion near-death rollouts, does a learnable prior's top-k (k≈12–24) **cover** the w128 survival/non-topout branches? If recall@24 < the w128 beam's own top-24 survival slice → the hedge is an irreducible breadth/coverage property, the moat is information-theoretic, **STOP the leapfrog**.
- **0b — clean-seed low-width G_π for the *actual Gumbel operator*:** if `G_π < 0.55` (CI-lower) at w4–w16 → no search gain at deployable width, leapfrog falsified, fall to the value-only escape hatch. **Do NOT inherit the 0.900** — it is triply invalid (contaminated seeds, beam-not-Gumbel, certifies a non-deployed head).

Five of six lenses converge here once their over-claims are stripped: authorizing a full MCTS self-play campaign *before* this probe repeats the exact over-claim pattern the campaign has been correcting six times.

---

## Rejected alternatives

- **MuZero / Stochastic MuZero** — a learned dynamics net / VQ afterstate codebook solves a problem we do not have (we own the exact cheap simulator; chance is enumerable). Strictly dominated — it injects model error into the one part we have exactly. *This is the one point of full six-lens convergence.*
- **Vanilla AlphaZero / PUCT visit-count targets** — fact (d) collapses the target to the starved near-one-hot beam root.
- **Value-only better-evaluator (status quo)** — fact (b) proves a better leaf alone ties at w128d9; also fails R4.
- **Full AlphaStar league (main + 3 exploiters)** — 4× an already-tight datagen budget for a non-transitivity dimension the ≤6%-coupling / 1-D-Elo evidence says is shallow. *Kept as a tripwire, not a build:* champion-pinning at 10% of the self-play pool + a 1-D-Elo antisymmetric-residual monitor (build a league only if residual > 10%).

## Build-vs-reuse (per the check-existing-tools rule)

**Build the search + self-play native in Rust on the in-repo seam.** Decisive call: every JAX/PyTorch MCTS runtime (mctx, LightZero) assumes a learned model batched on-device and would force ~1e5 per-node FFI hops/game into Rust movegen — fatal, and it abandons the built ANE/BLAS eval path.
- **Reuse (Rust):** `beam.rs` SearchState fork / `commit_placement` (bag dealt once) / transposition / the Committed-afterstate+bag loop (the exact chance-node seam) / `root_scores`; the `Mind` Policy trait (written anticipating "a future MCTS policy on the same trait"); `tetr-valuenet` OnnxEvaluator (ANE) + the batched-BLAS InferenceServer for datagen.
- **Reuse (reference only):** mctx `gumbel_muzero_policy` + `qtransforms.py`, LightZero `gumbel_muzero.py` + ReZero reanalyze scheduler — as algorithm/math references and the Python-side trainer/reanalyze orchestrator. Diff the Rust port against mctx on toy states. Optionally fork `treant-gumbel` (Rust, MIT: Gumbel-top-k + Sequential Halving, but PUCT-below-root and no chance nodes — patch both).
- **Rust/Python boundary = per-GAME shard I/O (Arrow/npz), never per-node.**
- **Must build new:** (a) two-headed net = conv trunk + ~34-way policy head; (b) the policy-consuming search vehicle (R4 fix); (c) exact chance-node expectimax + survival-CVaR backup; (d) the SSL board-reconstruction aux head.

## Pre-registered kill criteria (beyond Gate-0)

- **Target-extraction STOP:** completed-Q's `value_scale`/`maxvisit_init` relocate the τ/N_eff fragility that (per the campaign's R1) turned policy targets into "entropy injection" that *regressed* the net even at G=0.900. If completed-Q targets regress the net despite healthy G → calibration is the wall; STOP the policy axis rather than tuning knobs a 7th time.
- **Deployment-parity STOP:** if the π-guided search at ≤~2000 net-evals/move loses the pre-registered pair-GSPRT vs probe-tp128d9 at matched ~100 ms/move → the brute-width moat is uncrossable at matched wall-clock. Pre-registered *relaxation*, not a patch: the north star relaxes to "match at 100 ms, beat only at a larger per-move budget."
- **Throughput STOP:** first-hour datagen < 150 games/hr AND ReZero reanalyze cannot lift effective data → starvation recurred; STOP or move to cloud before authorizing a campaign.
- **Value-collapse STOP:** held-out std(z_hat) < 0.15 before round-2 → leaf degenerated to reward-dominated glass-cannon; STOP until the SSL aux restores per-state value signal.

## The four unresolved tensions (what evidence settles each)

1. **Does the improved-policy signal exist at deployable width?** → Gate-0 (recall@k + low-width Gumbel G_π). *The load-bearing one.*
2. **Is width replaceable, or is it epistemic humility about the net's own value errors?** → net-belief-CVaR-value @ w32 vs CC2-SPEC_DECAY @ w32 (matched low width).
3. **Where does the signal live — search backup or inter-episode outcome distribution?** → the same low-width G_π probe (does search move the policy at deployable width?).
4. **Chance-node enumeration** — settled: all six agree, and it is correct.
