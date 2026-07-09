---
id: T16
title: Round driver — one resumable command per round
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: [T14, T15]
---

## Question

Port the `rounds.py` discipline to the frozen design: one resumable command per round (datagen → train → completed-Q transform → start-gates → promotion pair-GSPRT → ledger row), manifest memoization = rerun-is-resume. Honor the ratified BALANCED rigor: freeze the gate + kill criteria only; architecture floats with a one-line amendment log (not a per-deviation contract). Round ABORTS (not slips) on the throughput STOP (<150 games/hr first hour) or start-gate failure. Incumbent = last PROMOTE row. Not-inherit the rl-branch infra bugs (SPRT verdict latch — already fixed in master's `sprt.rs`; seed-region plumbing; non-atomic shard writes).

## Round-1 amendment log (balanced-rigor contract; written before running)

- **A-r1-1:** round-1 datagen = 100% mirror self-play (no champion-pinning yet — needs two-arm datagen; round-2 item). Seed region 2,000,000+ (disjoint: 100k dev / 1M round-0 / 8xxM duels / 900M throughput probes).
- **A-r1-2:** the round-1 incumbent/datagen driver is round0_v3 @ epoch 1 (training interrupted at 2/3 epochs; slot head learning healthily, sCE 21.8→4.24; policy ≈ v2's epoch-1 level). Acceptable because the round retrains from scratch on new data; the incumbent only drives datagen + gates.
- **A-r1-3:** round-1 training mixes round-0 + round-1 corpora via a symlink dir (replay, ~orig design's spirit) and uses the live-logit reanalyze targets.
- **Round-1 promotion read (pre-registered):** `gate --a guided:round1@m12w8d5 --b guided:round0_v3@m12w8d5` (p1=0.55 latched) + G_π preflight for round1. Promote only on gate PASS.

## Round-1 execution log

- Datagen: serial run measured ~160 games/hr real (the 4-game probe's 1,370/hr was a short-game artifact — the T03 game-length confound again); **built `--workers` parallel fan-out (committed)** → **1,900 games in 22.8 min = 5,000 games/hr (6 workers)**. Round-1 corpus = 17 serial + 66 parallel shards, ~83k decisions (round-1 mirror games are SHORT, ~72 plies vs round-0's ~292 — fast decisive kills under rain).
- **Seat-A skew found**: 1059-841 (~5σ) in fixed-opener mirror games — z-label noise (obs don't encode seat). Fixed for future rounds (opener staggered by game parity, committed); round-1 data kept (noise, not artifact).
- Training: live-logit reanalyze mode on the 203-shard replay mix (round-1 + every-4th round-0 shard), 3 epochs, running. N_eff read on the first (round-0) shard = 6.68, marginally above band — the live form sharpens with logits; watch epoch metrics.
- Next: G_π preflight (`duel guided:round1@m12w8d5 vs policy:round1`) → promotion gate (`gate guided:round1 vs guided:round0_v3`, p1=0.55 latched, fresh seeds 920M+).

- **Round-1 training done** (203-shard mix, live-logit, 3 epochs): holdout pCE 2.009→1.692→1.717 (strong policy learning, mild epoch-2 uptick — epoch-1 was best but per-epoch exports overwrite; gate runs on epoch-2), sCE 4.01→3.96, **z_std 0.106/0.124/0.098 — BELOW the 0.15 VALUE-COLLAPSE STOP.** The pre-registered stop FIRES for round-2 datagen: mirror self-play between equal nets makes z weakly predictable (≈coin-flip outcomes) and the value head flattens — exactly the R3 mirror-flattening the kill criteria anticipated. **Round-2 may not start datagen until the value signal is restored** (the frozen design's SSL board-reconstruction aux head is the prescribed fix; value-bootstrap from stored root scores + A3-style per-source value weighting are the alternatives). Round-1's own promotion gate still runs (the stop governs round-2 datagen).

## Round-1 VERDICT: NO PROMOTION — and the root cause is isolated

- **Promotion gate:** `guided:round1 vs guided:round0_v3` = **H0Accepted at 33 pairs (llr −2.97)** — the candidate is not better (v3 swept 6 pairs to ~1). The first compounding attempt did not compound.
- **Diagnostic (policy-vs-policy, no search): 0-64.** The round-1 POLICY collapsed outright — far beyond value flattening. All-topout in seconds.
- **Root-cause analysis:** my live-logit target implementation is **theoretically unsound**: it re-mixes the *stored old* search Q with the *trainee's* ever-changing logits each batch → π' chases `logit + c·qnorm` → a self-amplifying runaway (no fixed point; entropy collapse; trunk distortion also explains z_std 0.098). Gumbel-MuZero's reanalyze form uses the **generator's frozen logits** (the net that ran the search), not the trainee's — I conflated them. The z_std VALUE-COLLAPSE stop caught the same trunk damage from the value side.
- **Controlled A/B running (Run A):** same 203-shard mix, STATIC targets (logits=None — the proven round-0 recipe). If Run A gates PASS/even vs v3, live-mode is condemned as the sole cause and round-2 proceeds with static (or frozen-generator-logit) targets; if Run A also collapses, the mirror-data mix is implicated too and the champion-pinned pool (A-r1-1 deferral) becomes load-bearing.
- **Method note:** the failure was caught by the pre-registered gate + a 9-second diagnostic, cost one training run, and produced a precise theoretical lesson — the balanced-rigor contract working as designed.

## Round-1 postmortem CORRECTED: the real cause was a datagen index-misalignment bug (mine)

Run A (same mix, STATIC targets) ALSO collapsed 0-64 → live-logit mode was NOT the operative cause (both arms shared the data). Hunting the data found it: with the placement filter active, `root_scores()` yields the FILTERED roots (~12) but `play_decision` wrote them BY INDEX into the FULL `hold_placements` list (~68) — wrong placements got wrong Q, the rest death-coded, and `placements[argmax]` PLAYED the wrong move (the round-1 games themselves were garbage — also explains the short games and possibly the seat skew). Round-0 (no filter) was aligned; guided DUELS were always correct (the Mind path aligns internally); only the driver's external re-derivation misaligned.

**Fix (committed):** placements+scores now derive from `beam.root_scores()` directly — aligned by construction; the record stores the beam's actual (filtered) roots, which is also the honest Gumbel-style subset target. Regression test pins served-children ≤ top_m and played == argmax-of-served.

**Record corrections:** (1) the "live-logit unsound" theory is UNTESTED (stays quarantined pending a clean A/B on uncorrupted data — the frozen-generator-logit form is still the right design either way); (2) the z_std collapse read is also confounded by the corrupt data — the VALUE-COLLAPSE stop may not fire on a clean round-1; re-measure. (3) the ~5σ seat skew may have been the misplay artifact — the parity fix stays (harmless) but the skew needs a clean re-read.

**Round-1 RERUN (clean): fresh corpus seeds 2010000 (workers 6) → static-target retrain → gate.**

## Round-1 CLEAN verdict: H0Accepted — a real negative result

Clean rerun (fixed driver, fair seat split, healthy training, z_std 0.151 ≥ gate): promotion gate **H0Accepted at 36 pairs (llr −2.95; v3 swept 16 pairs to 2)**; policy-vs-policy 5-59. **The loop as configured does not compound.** The corrupted-run artifacts are all explained (skew gone, value gate passes), so this verdict stands.

**Structural diagnosis (from all receipts):** the venue is strength-mismatched for self-play. Rain-8 was calibrated on CC2-strength bots (~292 decisions/game mirror); v3-guided mirror games last ~72 — the venue is effectively ~4× harsher relative to the agent, so self-play data concentrates on early-game quick-death states with compressed skill expression (echoes the graveyard's escalation-attenuation and mirror-decisiveness facts). Secondary suspects: subset-group policy targets lose full-set contrast; from-scratch-on-mix discards the r0 optimum.

**Round-2 (ONE change, pre-registered): fine-tune from the round0_v3 checkpoint** (`--init`) instead of from-scratch — the standard AZ loop form and the direct counter to "from-scratch-on-mix < from-scratch-on-r0". Preflight: 1 epoch + a cheap policy duel before the full round. Queued for round-3+ if round-2 fails: (a) venue curriculum for datagen (weaker rain, matched to agent strength; evaluation venue stays frozen), (b) full-set contrast negatives, (c) A3 per-source loss weighting, (d) champion-pinned pool.

## Round-2 verdict: policy compounds, vehicle doesn't — the bottleneck is the VALUE

- Fine-tune preflight (1 epoch, --init v3): **policy 40-24 over v3** (vs from-scratch's 5-59) — fine-tuning extracts real policy signal from self-play data. The first positive compounding link.
- **Promotion gate: H0Accepted (llr −2.98, 33 pairs — 29 SPLIT pairs, 3-1 sweeps).** At the deployed-vehicle level (guided w8d5) the two nets play nearly identical games: **search washes out the policy delta** (the leaf-compression phenomenon at w8d5). The policy head mainly picks candidates; search outcomes are VALUE-driven — and vCE has been flat (~0.68, z_std ~0.15 marginal) since round-0. The value head is the compounding bottleneck.
- **Round-3 (ONE change vs round-2): search-value bootstrap for the value target** — every decision's stored played-child root score is a d5 search value estimate (dense: per-decision, vs z per-game); regress z_hat toward tanh(score/Z_SCALE) alongside the z CE (frozen-generator signal — sound; the W-composition inside the score is training machinery, allowed; the deployed bot reads only the WDL head). Evidence: Phase-A distilled deep-search values to R²=0.638; conv_rb1 (the shipped NNUE) was trained this way. Queued after: SSL aux head (design-prescribed), venue curriculum, contrast negatives.

## Round-3 verdict: value bootstrap FAILED-AS-IMPLEMENTED — the score-scale trap (C6) in a new form

Value-isolation duels (`value:` arm, d1 argmax — the clean value-head read, 64 games each):
- round3 (bootstrap) value **2-62** vs v3 — the bootstrap actively destroyed the value head.
- round2 (plain fine-tune) value **12-52** vs v3 — even plain fine-tuning on the mix degrades the value head.
- Root cause of the round-3 failure: `tanh(score/Z_SCALE)` assumed net-contract units, but the **round-0 shards' scores are CC2-eval units** (−37k…−1.4k) — on 68% of the mix the bootstrap target was a wrong-scale, systematically-pessimistic warp. This is the prior campaign's pre-registered **value-scale trap / C6 (scores don't transfer across scales)** re-hit in a new place. Shards do not record the generator eval's scale — that metadata gap is the enabling defect.
- Round-3 policy also dipped (30-34 vs round-2's 40-24) — shared-trunk gradients from the bad value target.

**The compounding scoreboard after three rounds:** policy fine-tuning extracts real signal (40-24); the value head is the bottleneck and NO variant has improved it (round-0-pure v3 value is best-in-class); vehicle gates split (search washes out small deltas). **Round-4 (ONE change vs round-2): the SSL board-reconstruction aux head** — the design freeze's prescribed value-signal densifier, scale-free by construction (reconstructs the own plane; no score units anywhere). Also queued: `eval_scale` metadata in shards (fixes the bootstrap soundly); larger self-play volume (5-10k games now cheap).

## Round-5 verdict: per-SHARD A3 masking = catastrophic interference (0-64 both heads)

Long runs of policy-free gradients (r0 shards, 64% of data) let the shared trunk drift from the policy heads (train sCE exploded to 53; top1 0.098). The inherited A3 mixed per-ROW within 50/50 batches — per-shard sequential masking is not the same thing. The sound form needs cross-shard batch interleaving in the streaming trainer (queued, fresh build).

**Round ladder so far (all single-variable, all receipted):** (1) from-scratch/mix: policy 5-59. (2) fine-tune: policy 40-24 ✓, value 12-52 ✗, gate H0-split. (3) +boot-value: value 2-62 (C6 scale trap). (4) **+SSL: policy 34-30 ✓, value 23-41 (best fine-tuned value) — gate pending.** (5) +A3-by-shard: 0-64 (interference). Running round-4's gate now.

## Block synthesis (2026-07-08 late): five rounds, zero promotions, one clear constraint

Round-4's gate: **H0Accepted (llr −3.02; 23 split, 11-2 sweeps to v3)** — the best fine-tune candidate also fails at the vehicle level. The through-line of all five rounds: **the value head is the vehicle's engine, and nothing trained on the round-1 mix matches v3's round-0-pure value** — because the self-play data's value labels are structurally weak (mirror games ~4× shorter than CC2's under the same rain; z weakly predictable; skill expression compressed). Policy gains are real (best 40-24) but search washes them out at w8d5.

**The binding constraint is value-learnable self-play data, not the training recipe.** Round-6 candidates, ranked by evidence: (1) **venue curriculum for datagen** — weaker rain matched to agent strength → longer games → value-discriminating positions (the evaluation venue stays frozen; obs venue-clock features make this a measured domain-shift experiment); (2) **champion/CC2-pinned pool** (two-arm datagen — grounded, longer games); (3) row-level A3 with cross-shard batch interleaving; (4) volume + compounded fine-tune iterations (needs the T16 driver for cadence). Infra receipts this block: 5,000 games/hr datagen, ~1h round cycles (the old campaign never completed one), every failure diagnosed to a mechanism within the hour.

## MILESTONE + round-6 re-ranking (2026-07-08 night)

- **Rain-curriculum hypothesis REFUTED by probe:** game length flat (~16-17 dec/seat) at rain 8/12/16 — mirror games end by self-inflicted topout, not rain pressure. Venue tweaks are not the lever.
- **Teacher-anchor duel: `guided:round0_v3@m12w8d5` BEATS `beam:cc2@w8d5` 21-11** (0.656, 20/32 escalation-length games). **First measured head-to-head win of the fully-learned vehicle over the hand-tuned CC2 eval at matched search config** (modulo SPEC_DECAY still in the search — the known training carve-out). The incumbent is NOT weak; short mirror games are a style property (both sides fast-killing), not a strength deficit.
- **Round-6 = two-arm datagen (grounded-opponent pool):** v3-vs-CC2 games are long, competitive, and value-rich — exactly the missing value-learnable distribution, and exactly what the gates measure. Build: datagen driver plays net-arm vs cc2-arm (seat-alternating), shards record BOTH seats (the net seat's rows train as usual; the CC2 seat's rows are r0-grade grounded data). This was A-r1-1's deferral, now evidence-promoted to the lead.

## Round-6 halted: the datagen driver has a THIRD defect (seed-matched divergence from the harness)

- Two-arm corpus came out absurd (net ~1200-0 over attack-tuned CC2 TP-beam; trusted duel says 9-7 ≈ even). Found + fixed defect #2: **tie-breaking** (driver used last-maximum; the planner's rule is FIRST-maximum, and CC2 ties on ~55% of decisions — the "CC2" seat played pathological last-tie moves). Committed with the fix.
- After the fix, still wrong: **seed-matched experiment** (same `guided:v3` mirror arms, same seeds): duel harness ≈ 88s/game; my driver ≈ 70 decisions/game in 5s. The driver structurally produces shorter/weaker games than the controller harness for identical inputs. Even round-0's CC2-mirror (~146 plies/game) was ~2-3× shorter than harness CC2 mirrors (escalation-length).
- **Scope of damage:** all datagen corpora carry this behavioral skew (round-0's teacher was "driver-CC2", not harness-CC2). Internal comparisons between nets trained on consistent data retain validity (the round ladder's relative reads stand), and ALL duel/gate results are trusted-harness (the 21-11 milestone stands). But the loop's data plant must match the harness before any further rounds.
- **Next probe (designed):** instrument one seed — log per-ply (placement pose, board height) in both the driver and a harness-driven game; diff to the FIRST divergent ply. Candidate mechanisms: gravity/think-time frames absent in the driver (pieces placed from spawn height with dt=0 while controllers pass real dt frames), replay/desync subtleties, spawn-wait differences.

## The TRUE (slot) vehicle's first harness reads (post-fix, 2026-07-09)

- **Slot-guided BEATS the full beam 12-4** at matched w8d5 (same net leaf) — the learned prior's top-12 restriction actively out-plays unrestricted width: learned selectivity > brute breadth, the leapfrog mechanism measured. (The per-child vehicle only drew 6-6/8-8.)
- **Slot-guided beats the CC2 teacher config 24-8 (0.75)** (vs per-child's 21-11) — the strongest fully-learned bot yet.
- T16 round driver BUILT (`python/tetrnn/round.py`): one resumable command per round; encodes all five-round lessons; JSON ledger. Round-6 = the first fully-consistent round (slot vehicle end-to-end, grounded two-arm data, fine-tune+SSL).

## Round-6 (first fully-consistent round): H0, but the VALUE SIGN FLIPPED

Ledger: gate H0Accepted (55 pairs, llr −3.04 — a longer, closer race than rounds 1-5); **value duel 31-17 FOR the candidate** (first value improvement of the campaign; prior best 23-41 against); policy 23-25 parity. The grounded two-arm data + fine-tune + SSL recipe fixed value-learnability as designed — per-round deltas are just sub-threshold for a p1=0.55 vehicle gate.

**Amendment A-r7 (logged, balanced-rigor): training LINEAGE decouples from PROMOTION.** Each round fine-tunes from and generates data with the NEWEST net (AZ-standard continuous training); the INCUMBENT (gate opponent/deployment candidate) advances only on gate PASS (pre-registered strictness preserved). Driver updated (`--lineage`). Round-7 chains from round-6's net.

## ⭐ ROUND 7: H1Accepted — THE FIRST PROMOTION (2026-07-09)

`gate | H1Accepted after 74 pairs (llr +3.05)` — the round-7 net (lineage: v3 → r6 → r7, each round = fresh grounded self-play + fine-tune + SSL, static completed-Q targets) **beats incumbent round0_v3 at the vehicle level** under the pre-registered p1=0.55 latched gate. Value duel 30-18 (second consecutive value win); policy 20-28 (the value is carrying it — consistent with search-outcomes-are-value-driven). **The expert-iteration loop compounds.** The A-r7 lineage decoupling was the unlock: single-round deltas are sub-threshold, but two chained rounds cleared the strict gate.

**Incumbent advances: `r7/net` is the new incumbent.** Round-8 launched (lineage = incumbent = r7). Next anchors for the promoted net: CC2 teacher (was 24-8 for v3), the champion ladder (tp:cc2@w16d7 upward), solo (baseline 0.0 — unchanged axis).

## Round 7 promotion VOIDED by anchor evidence — the non-transitivity tripwire fired

Anchors for the "promoted" r7: **0-32 vs beam:cc2@w8d5** (v3 scores 24-8) and 0-32 vs tp:cc2@w16d7, all instant-topout games. r7 genuinely beats v3 head-to-head (the gate was честный) — it evolved a **parent-exploiting glass cannon** (attack-rush that kills its ancestor before its own stack collapses; CC2 downstacks the rush). Classic self-play strategy collapse — the design's pre-registered non-transitivity tripwire, fired for real. **Incumbent rolled back to round0_v3.**

**A-r8 (committed):** (1) promotion now requires the gate PASS **AND** anchor no-regression (guided candidate ≥18/48 vs the fixed CC2 anchor — v3 scores ~36/48); the anchor duel joins every round's telemetry. (2) Self-play pool diversified: half grounded-vs-CC2, half mirror (a homogeneous pool bred the exploit). Round-8 relaunched: lineage r7 (momentum retained — the anchor gate now catches degenerates), incumbent v3.

## Forensic: the degeneracy began at ROUND-6 (r6 anchor = 0-32)

`guided:r6/net` vs `beam:cc2@w8d5`: **0-32** — the first fine-tuned round was already the glass cannon; the celebrated 31-17 "value improvement" vs v3 was part of the degenerate direction (hypothesis: the grounded corpus's net-seat wins were largely rush-kills → z=+1 taught rush=win; the value head learned to love rush states). Mechanism hypotheses, in test order: (1) homogeneous grounded pool (already fixed, A-r8); (2) fine-tune LR 1e-3 too hot for a 1-epoch delta (standard fix: 1e-4); (3) SSL/aux gradient mix. **Decision rule:** round-8 (lineage r7, diversified pool) faces the anchor veto — if voided, restart lineage from v3 under the hardened methodology; if THAT degenerates, drop the fine-tune LR.

## Round-8: H1_VOIDED_BY_ANCHOR(0/48) — the hardened gate works autonomously

The r7-lineage candidate again "beat" v3 at the SPRT gate (llr +2.96, 103 pairs) and scored **0/48 vs the CC2 anchor** → auto-voided by the A-r8 rule, no human in the loop. Diversified data did NOT rehabilitate the exploit lineage. Per the pre-registered decision rule: **lineage restarts from round0_v3** (round-9, launched); if the v3-restart also fails the anchor, the fine-tune LR (1e-3) is the next suspect (drop to 1e-4).

## Round-9: H1_VOIDED_BY_ANCHOR(0/48) — the v3-restart degenerates in ONE round → the LR is the suspect

Fresh lineage from v3, diversified pool, one epoch of fine-tune at LR 1e-3: policy 14-34, value 9-39, **anchor 0/48**, and STILL "beats" v3 at the gate (llr +2.98) — the fourth consecutive incumbent-beating/anchor-failing candidate. Per the decision rule, the training step itself is now implicated: **LR 1e-3 on a 1-epoch fine-tune rewrites the policy wholesale toward the exploitable patterns of the very opponent whose games are in the corpus** (the mix's grounded half contains the incumbent's own play — beating the incumbent while failing the anchor is opponent-overfitting, not strength). **A-r10: `--lr` flag added; round-10 running at 1e-4** (the small-delta regime). If 1e-4 holds the anchor but gains nothing, the schedule between (LR × epochs × data share) becomes a measured sweep.

## ⚠️ INSTRUMENT FORENSIC (2026-07-09): the r6-r10 verdicts were VEHICLE artifacts — the hidden ranker chooser

Round-10 (LR 1e-4) anchored 0/48 — implausible for a near-clone of v3 — so the anchor instrument itself went under the microscope. Findings, in evidence order:

1. **The control regressed**: `guided:round0_v3@m12w8d5` vs `beam:cc2@w8d5` on seeds 992000000 read **24-8 at ~19:40 Jul 8** and **0-16 (7-second games) tonight** — same weights (safetensors/config sha-verified untouched since the 09:01 export), same seeds, same CLI string.
2. **Bisect exonerated the beam levers**: a worktree at dcb36f3 (pre-levers #1/#3) also reads 0-16.
3. **Root cause: `guided_filter` was a hidden `has_slot_head()` chooser** (eb39302, ~19:33 Jul 8). The moment v3's export carried a slot head, every rebuilt binary silently swapped the vehicle from the per-child policy ranker to the slot ranker — under the SAME arm string. The 24-8/12-4 "slot vehicle" reads at ~19:40 almost certainly ran a STALE binary (still per-child): the arm fix was committed without a rebuild of the duel binary.
4. **The slot ranker collapses play**: v3 slot-guided anchors **0-16 with instant-suicide games** (control B, explicit `sguided:`). The slot head (sCE plateaued ~4.0 after 2 epochs) is far too weak to filter every node.
5. **Datagen was equally contaminated** (`datagen.rs` uses `guided_filter`): rounds 6-10 generated training data with the suicide vehicle AND measured anchors with it.

**Consequences:**
- **ALL r6-r10 verdicts are VOID** — not because the gates lied about net-vs-net, but because the vehicle underneath datagen + anchor was the degenerate slot ranker. The "glass cannon lineage" (A-r8) and "LR 1e-3 too hot" (A-r10) narratives are UNPROVEN — they explained artifacts of the wrong vehicle. The A-r8 anchor-veto mechanism itself remains sound and stays.
- **Fix (landed)**: the ranker is now EXPLICIT in the arm grammar — `guided:` = per-child (validated), `sguided:` = slot (experimental until the slot head trains stronger). `guided_filter` no longer chooses. No hidden dispatch on model contents may ever select a vehicle again.
- **Lesson (twice now)**: silent behavior swaps under an unchanged interface string are instrument death. Arm strings must pin EVERY behavior-relevant choice.

## ROOT CAUSE UNDER THE FORENSIC (2026-07-09): the slot head served a ReLU-DEAD trunk — child-only whitening

Why was the slot head near-uniform (held-out hit@12 0.33 vs random 0.20 vs per-child head 0.996)? Not undertraining. The frozen-trunk refit instrument (`tetrnn/fit_slots.py`) came back **grad-norm 0.0, embedding std 0.0**: the trunk outputs ZERO on every parent observation. The current-piece one-hot (feature dims 43-49) is constant-zero in CHILD rows (post-placement) but set in PARENT rows; whitening stats were computed over children only, so those dims' std floored to 1e-6 and parent rows standardized to **z = 1e6**, saturating every ReLU. The slot head trained AND served on a dead trunk — the only thing it could learn was marginal slot popularity, i.e. a **state-blind constant filter** (the same 12 slots at every node) → the suicide play. This is the constant-feature landmine (first seen in the ONNX-parity saga) firing in production, exactly where predicted.

**Fixes landed:** whitening stats now stream over children AND parents (train.py; doc-block carries the forensic + the remaining opp/venue landmine note). `fit_slots.py` kept as a permanent instrument (frozen-trunk slot refit + hit@k). **round0_v4 retraining now** (same corpus/recipe, union whitening, full 3 epochs — v3-as-shipped was accidentally epoch-1-of-3; both deltas noted). Validation battery for v4: duel vs v3 (per-child), CC2 anchor (v3 reference 9-7/16 on 992M), slot hit@12 (bar: approach per-child's 0.996), and — if the slot head qualifies — `sguided:` anchor + parity, restoring the fast vehicle and ~1h rounds.

**Vehicle economics measured (2026-07-09):** per-child datagen = 112 games/hr @2 workers BLAS (≈330/hr @6 ≈ 3.6h/round for 1200 games); TETR_ORT CoreML is SLOWER here (80/hr — single-state filter forwards are ANE-latency-bound). The slot vehicle (one forward/node) is the only structural path to fast rounds — hence the retrain-first order.

**Control receipts (2026-07-09, explicit-ranker binary 0cebd90):** `guided:` (per-child) v3 vs `beam:cc2@w8d5` on 992M = **9-7 over 16 full-length games** (375s); `sguided:` (slot) same seeds = **0-16 in 15s** (instant suicide). The original 24-8 was 16 pairs (32 games) — the 8-pair control's 9-7 is compatible with it under sampling noise, but the run's receipt can't prove which ranker the stale binary ran (spec.json recorded the TREE's commit, eb39302+dirty, not the binary's; games.jsonl is unflushed/empty). Fixed forward: receipts now embed the BINARY's build commit (`build_commit`/`build_dirty` in spec.json). v3's true anchor strength = somewhere in the 9-7/32-ish..24-8/32 band; the per-child vehicle is competitive either way, and v4's anchor read will supersede it.

## PRE-REGISTRATION (2026-07-09, before v4 data): the v4 gate + round-11 decision tree

**v4 validates as the new base** iff (b) `guided:v4` vs `guided:v3` is not clearly worse (≥6/16-ish; they share corpus+recipe — parity expected, the whitening fix should only ADD parent-trunk life) AND (c) v4's CC2 anchor lands in/above v3's 9-7..24-8 band. If v4 clearly regresses on either: suspect the schedule delta (3 full epochs vs v3's accidental 1) — fall back to the epoch-0 snapshot (`round0_v4_e0`) and re-read (b)/(c) before any deeper surgery.

**Slot vehicle qualifies for round-11** iff (a) hit@12 materially closes on the per-child bar (≥0.85-0.90; v3's dead-trunk read was 0.34, per-child 0.996) AND (d) `sguided:v4` anchors ≥ parity-band vs CC2 AND (e) sguided-vs-guided parity is not a collapse. Qualified → round-11 runs `--vehicle sguided` (fast rounds, ~13 vs ~73 forwards/node). Not qualified → `--vehicle guided` at 1200 games (~4-6h/round, measured 112 games/hr @2w) and slot rehab continues in parallel (fit_slots on the live trunk / m=16-24 dial as fallback).

**Round-11 verdict tree (recipe: driver defaults, lineage=incumbent=validated base, LR default 1e-3):**
- gate H1 + anchor ≥18/48 → the campaign's first honest PROMOTION.
- gate H0 + anchor healthy → chain lineage (A-r7), run rounds 12-13; reassess the compounding hypothesis only after ≥3 consistent-vehicle rounds.
- anchor FAIL → the incumbent-exploit mechanism is REAL (not a vehicle artifact) — then and only then re-test LR 1e-4 (A-r10's clean shot) and the pool-composition levers.

**v4 epoch-0 early read (2026-07-09): the whitening fix REVIVES the slot path.** Parent trunk std 1.37 (v3: 0.00 — dead); slot hit@12 **0.816** / hit@16 0.874 / hit@24 0.937 after ONE epoch (v3's dead-trunk head: 0.34/0.42/0.60). Epochs 1-2 pending; the m16/m24 dial (~4.5×/3× cheaper than per-child) is the fallback if hit@12 plateaus under the 0.85-0.90 bar. Epoch-0 snapshotted (`round0_v4_e0`).
