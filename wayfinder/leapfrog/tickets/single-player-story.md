---
id: T10
title: Single-player from the same system
labels: [wayfinder:grilling]
status: open
assignee:
blocked-by: [T04, T08]
---

## Question

The destination demands the best bot in BOTH settings from one learning system. Decide how solo emerges:

- **Multi-task training** (solo marathon/downstack as additional venues with their own returns), **conditioning** (venue features already exist in obs — extend?), or **post-hoc finetune** of the versus net?
- What is the solo objective without hand-tuned reward — survival + real score? Attack? (Solo APP is combo-farmable and gate-only; the E8 confound says attack-only values are suicidal under rain.)
- Does the solo bar (champion 0.8225 APP held-out, downstack battery) need its own search budget rules?

Output: the solo training + gating plan, consistent with the frozen design and the gate battery.

## First measured solo baseline (2026-07-09): the versus vehicle scores ZERO

`solo` subcommand added (marathon-holdout convention: 16 VALIDATION seeds, cap 150). Reads:
- **`guided:round0_v3@m12w8d5`: mean APP 0.000, topped 16/16** — the versus-dominant vehicle (24-8 over its CC2 teacher) stacks itself to death on an empty board with no pressure. Versus training does not transfer to solo AT ALL (the policy learned pressure-conditioned patterns that self-destruct without rain/opponent).
- `value:round0_v3` (d1): APP 0.225, topped 7/16 — the value head alone is less degenerate than the full vehicle.
- Champion reference on the same battery: **0.787** (tp:cc2@w128d9, 0 topouts).

T10 is a genuine second training axis, not a freebie: candidate designs (multi-task solo datagen venue + mixed training; solo z definition — survival/score; venue-conditioned obs) now have a concrete 0.0 baseline to beat.

## PROPOSED design (2026-07-09): solo as a CRN-paired self-race — one system, no new machinery

The versus loop's whole apparatus (two-seat datagen, terminal-WDL z, completed-Q targets, the gate battery) transfers to solo UNCHANGED if solo is framed as a **race**: two independent boards, same piece seed, **no garbage exchange, no rain** (venue features already encode this), z decided at the cap by env-truth:

- topped vs survivor → survivor wins (z ±1)
- both topped → later topout wins (survival pressure)
- both survive to cap → more total attack wins; equal → draw (z 0)

Why this is the right shape:
- **Purity holds**: z is environment truth (survival order, attack totals) — no hand reward, no APP in the objective. APP stays a gate-only holdout metric, exactly as the versus side treats it.
- **It optimizes the actual solo skill** — attack efficiency under self-survival — not bare survival (trivially maxed by low stacking) and not raw attack (the E8 suicide confound doesn't apply at rain 0, but topping out still loses the race, so survival is priced in).
- **Zero new machinery**: `datagen_game_vs` with the interaction switched off + a venue flag; the trainer, targets, shards, gates are untouched. The net is venue-conditioned through existing obs features, so ONE net serves both modes (the destination's "same system" demand).
- **CRN pairing for free**: same-seed boards make the race a paired comparison — low-variance z, the same trick the duel instrument uses.

Decision experiments (in order): (1) does mixed training (versus + solo-race shards) hold the versus anchor while lifting solo APP off 0.0; (2) mix ratio sweep only if (1) regresses; (3) venue-feature ablation (do the clocks/rain dims suffice for the net to separate the modes). Bar: solo APP > 0 and climbing toward the champion refs (0.787@150 / 0.8194@600) without versus-anchor regression.

Blocked behind: the versus loop compounding honestly (round-11+, post-v4). This is the design-on-the-shelf so the axis starts the day the loop closes.
