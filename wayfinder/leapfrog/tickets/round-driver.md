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
