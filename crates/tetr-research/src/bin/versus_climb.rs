//! Versus weight climb: hill-climb CC2 board params against a fixed incumbent
//! under the rain format — the win-rate-climb instrument the roadmap calls for.
//!
//! Method: a (1+1)-ES with common random numbers. Each candidate plays the
//! incumbent on the SAME train seeds in BOTH orientations (arm swap), scored by
//! a dense paired objective: death verdict (±1000 — death is what decides
//! matches) plus the net-attack margin (the cap-game tiebreak). Proposals
//! jitter each param relative to its magnitude (plus a small absolute floor so
//! near-zero params can move); sigma adapts by an accept-rate rule. The climb
//! is fully reproducible from CLIMB_SEED, and self-bounded by TIME_BUDGET_SECS.
//!
//! A held-out validation (disjoint seeds) reports the honest verdict at the
//! end: wins by death, cap tiebreaks, and mean margin — climbed weights ship
//! only if validation clears, never on the train objective.
//!
//! RUN RECORD (2026-06-10, 1h, defaults, rain 0): train objective 0 → +376.4
//! over 127 iters / 7 accepts — and validation came back DEAD EVEN (deaths
//! 19-19, tiebreaks 11-14, margin −0.9): pure seed overfitting from reusing 24
//! CRN seeds across every candidate. The gate worked; nothing shipped. Two
//! durable facts from the run: the paired objective is exactly fair (the
//! identical-weights baseline evaluates to +0.0), and candidate-vs-incumbent
//! matches are ~59% death-decisive even WITHOUT rain (asymmetric styles kill;
//! only same-weights mirrors are bland) — the real format has live death
//! signal. Next run's regularization: more/rotating train seeds, an
//! acceptance bar above noise (re-evaluate the incumbent or SPRT per
//! candidate), and a periodic held-out check during the climb.
//!
//! RUN RECORD v2 (same day, ROTATE=1, ACCEPT_MARGIN=25, SIGMA=0.15): rotation
//! killed seed memorization but the bar was far below the objective's noise —
//! with ±1000 death spikes a 48-match block mean has σ ≈ ±90, so 25 let lucky
//! proposals through (20 accepts / 56 iters), the accept-rate rule then GREW
//! sigma, and the parameters random-walked (tslot3 4.9 → 17.3); validation:
//! deaths 14-16, tiebreaks 8-57, margin −14.8. Gate caught it again; nothing
//! shipped. CALIBRATION LESSON: ACCEPT_MARGIN must be ~2σ of the block mean
//! (~150-200 at these sizes), with SIGMA small (~0.08) so a rare false accept
//! cannot teleport the walk.
//!
//! RUN RECORD v3 (same day, ACCEPT_MARGIN=150, SIGMA=0.08): the calibrated
//! design behaved exactly as intended — 45 iters, ONE accept (+213.9 over a
//! +0.0 incumbent block), and the first POSITIVE held-out validation: deaths
//! 20-15, tiebreaks 32-28, margin +0.79. Not statistically significant
//! (20-15 of 35 decisive ⇒ p ≈ 0.25 one-sided), so NOT shipped — but the
//! candidate is a small, sane perturbation of attack_tuned worth a long SPRT:
//! [-0.0036628882, -1.5733862, -0.19578815, -0.34977585, -1.5387586,
//!  -5.149458, 0.3575636, 0.09665186, 1.550793, 4.4781384, 3.782923]
//! SPRT EPILOGUE (same day, `versus_sprt`): the v3 candidate was REJECTED —
//! H0 accepted in 270 s (266-269 of 544 decisive, LLR −2.99). The one accept
//! the calibrated climb produced in an hour was noise after all; the gate
//! chain (held-out validation → SPRT) worked end to end and nothing shipped.
//! Honest read: board-params-only jitter around attack_tuned has no cheap
//! survival gains at this match budget — further wins want a different lever
//! (re-priced garbage-aware weights, deeper search, or the NN round), not
//! more of this walk.
//!
//! THE CONFIRMER (same day, post-epilogue): the rotate path now runs exactly
//! that design — a screened accept must additionally win an SPRT race
//! ([`tetr_research::sprt`], proposal vs the walk's current incumbent, a
//! fresh per-iter region, capped by CONFIRM_MATCHES) before it moves the
//! walk; H0 or in-budget inconclusive DEMOTES it. Block means at these
//! sizes pass noise (v1/v2/v3 all proved it); the racer does not.
//!
//! THE ANCHOR (2026-06-11): per-accept confirmation bounds each STEP's
//! false-accept rate (CONFIRM_ALPHA, default 0.02), but a long walk takes
//! many steps and the per-step α accumulates — eventually some confirmed
//! accept is noise, and the walk ratchets on an illusion. So every
//! ANCHOR_EVERY confirmed accepts, the walk must additionally beat its last
//! ANCHORED point (the last SPRT-verified composition of accepts) in a fresh
//! race: H1 re-anchors at the current params, H0 ROLLS the walk BACK to the
//! anchored point, inconclusive keeps the old anchor and retries after the
//! next ANCHOR_EVERY accepts. Drift between anchors is therefore bounded by
//! one anchor window, and everything past the last anchor is always
//! SPRT-verified end-to-end — alpha accumulation buys noise for at most one
//! window, never the campaign.
//!
//! CAMPAIGN REGIONS + RESUME (2026-06-11): validation, rotation,
//! confirmation, and anchor seeds now come from the CAMPAIGN's private slab
//! ([`tetr_research::seeds::Campaign`]) — fresh per campaign name, so
//! repeated campaigns stop iterating against one static validation region.
//! Train (the non-rotate path) stays at `regions::TRAIN`. Every run writes a
//! [`tetr_research::ledger`] run directory (spec, per-iteration outcomes,
//! checkpoint each iteration, summary); RESUME=<run-dir> continues an
//! interrupted walk bit-identically from its checkpoint (state carries the
//! RNG word, sigma, iteration, params, and anchor bookkeeping — witnessed by
//! the `resume_reproduces_the_uninterrupted_walk` test). All pre-campaign
//! trajectories above (v1/v2/v3/confirmer) reproduce only at pre-move
//! commits; their verdicts stand.
//!
//! Env: TIME_BUDGET_SECS (1800), CAMPAIGN ("scratch"), RESUME ("" — a prior
//!      run dir to continue), MAX_ITERS (0 = unbounded, bounds THIS
//!      invocation), SEEDS (24 train; the per-iter block size when rotating),
//!      VAL_SEEDS (32), ROTATE (1), ACCEPT_MARGIN (25), RAIN_PERIOD (8),
//!      MAX_PLIES (240), BEAM_DEPTH (2), BEAM_WIDTH (16), SIGMA (0.15),
//!      CLIMB_SEED (1), CONFIRM_MATCHES (800; 0 disables the per-accept
//!      confirmer — rotate path only), CONFIRM_ALPHA (0.02), ANCHOR_EVERY
//!      (3; 0 disables anchoring), ANCHOR_MATCHES (800).

use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::json;

use tetr_core::ai::Cc2Weights;
use tetr_research::bots::BotSpec;
use tetr_research::cli::{SplitMix64, env_f64, env_string, env_usize};
use tetr_research::ledger::RunLedger;
use tetr_research::seeds::{Campaign, seed_set, seed_set_from};
use tetr_research::sprt::{SprtConfig, SprtReport, SprtVerdict, sprt_race};
use tetr_research::versus::{VersusFormat, evaluate_versus_format};

/// One standard-normal draw (Box-Muller over the deterministic SplitMix64).
fn gauss(rng: &mut SplitMix64) -> f64 {
    let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

type BoardParams = [f32; Cc2Weights::BOARD_PARAM_COUNT];

/// The paired objective of `candidate` vs the incumbent over `seeds`, both
/// orientations: mean over matches of `1000·(they died − we died) + (our net
/// attack − theirs)`. Higher is better; the death term dominates by design
/// (death decides matches; margin orders the rest).
fn objective(
    params: &BoardParams,
    incumbent_params: &BoardParams,
    width: usize,
    depth: u8,
    seeds: &[u64],
    format: VersusFormat,
) -> f64 {
    let cand = bot(params, width, depth);
    let incumbent = bot(incumbent_params, width, depth);

    let fwd = evaluate_versus_format(&cand, &incumbent, seeds, format);
    let rev = evaluate_versus_format(&incumbent, &cand, seeds, format);

    let mut total = 0.0f64;
    let mut n = 0u32;
    for o in &fwd.outcomes {
        let death = 1000.0 * (f64::from(o.b_topped) - f64::from(o.a_topped));
        total += death + f64::from(o.attack_a) - f64::from(o.attack_b);
        n += 1;
    }
    for o in &rev.outcomes {
        let death = 1000.0 * (f64::from(o.a_topped) - f64::from(o.b_topped));
        total += death + f64::from(o.attack_b) - f64::from(o.attack_a);
        n += 1;
    }
    total / f64::from(n.max(1))
}

fn bot(
    params: &BoardParams,
    width: usize,
    depth: u8,
) -> impl Fn(u64) -> Box<dyn tetr_core::player::PlayerController> + Send + Sync + 'static {
    BotSpec::beam(width, depth)
        .cc2(Cc2Weights::attack_tuned().with_board_params(params))
        .factory()
}

/// Everything the loop needs that does not change while it runs.
struct ClimbConfig {
    width: usize,
    depth: u8,
    format: VersusFormat,
    /// Per-iteration screen block size (rotate) / train set (non-rotate).
    train_seeds: Vec<u64>,
    rotate: bool,
    accept_margin: f64,
    confirm_matches: u32,
    confirm_alpha: f64,
    anchor_every: u32,
    anchor_matches: u32,
    budget: Duration,
    /// New iterations allowed in THIS invocation (0 = unbounded).
    max_iters: u32,
    campaign: Campaign,
}

/// The walk's complete resumable state — everything the next iteration reads.
/// A checkpointed state plus the same config reproduces the uninterrupted
/// trajectory bit-for-bit (the RNG is carried as its raw word).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ClimbState {
    schema_version: u64,
    campaign: String,
    iter: u32,
    accepts: u32,
    anchor_events: u32,
    accepts_at_last_anchor: u32,
    sigma: f64,
    rng_raw: u64,
    best_params: BoardParams,
    origin_params: BoardParams,
    anchored_params: BoardParams,
    /// Wall-clock seconds consumed by prior invocations (reporting only —
    /// each invocation is freshly bounded by TIME_BUDGET_SECS).
    consumed_secs: u64,
}

fn fresh_state(campaign: &Campaign, sigma: f64, climb_seed: u64) -> ClimbState {
    let origin = Cc2Weights::attack_tuned().board_params();
    ClimbState {
        schema_version: 1,
        campaign: campaign.id.clone(),
        iter: 0,
        accepts: 0,
        anchor_events: 0,
        accepts_at_last_anchor: 0,
        sigma,
        rng_raw: climb_seed,
        best_params: origin,
        origin_params: origin,
        anchored_params: origin,
        consumed_secs: 0,
    }
}

fn race(
    cand_params: &BoardParams,
    incumbent_params: &BoardParams,
    cfg: &ClimbConfig,
    seed_base: usize,
    max_matches: u32,
    alpha: f64,
    deadline: Instant,
) -> SprtReport {
    sprt_race(
        &bot(cand_params, cfg.width, cfg.depth),
        &bot(incumbent_params, cfg.width, cfg.depth),
        cfg.format,
        SprtConfig {
            alpha,
            seed_base,
            max_matches,
            deadline: Some(deadline),
            ..SprtConfig::default()
        },
    )
}

fn report_json(r: &SprtReport) -> serde_json::Value {
    json!({
        "verdict": format!("{:?}", r.verdict),
        "wins": r.wins,
        "losses": r.losses,
        "pairs": r.pairs,
        "llr": r.llr,
    })
}

/// Run (or continue) the walk until the time budget or MAX_ITERS ends this
/// invocation. Returns the final state and the exit reason.
fn run_climb(
    cfg: &ClimbConfig,
    mut state: ClimbState,
    mut ledger: Option<&mut RunLedger>,
) -> (ClimbState, &'static str) {
    let start = Instant::now();
    let deadline = start + cfg.budget;
    let block = cfg.train_seeds.len();
    // Stride ≥ the race's worst-case seed consumption (M matches use at most
    // M/2 indices; M covers it with slack), and never below the historical
    // 4096 so overridden caps cannot make successive races overlap.
    let confirm_stride = 4096usize.max(cfg.confirm_matches as usize);
    let anchor_stride = 4096usize.max(cfg.anchor_matches as usize);
    let mut iters_this_run = 0u32;
    // The non-rotate path's running best — re-derived deterministically on
    // resume (same params, same fixed train seeds).
    let mut fixed_best = (!cfg.rotate).then(|| {
        objective(
            &state.best_params,
            &state.origin_params,
            cfg.width,
            cfg.depth,
            &cfg.train_seeds,
            cfg.format,
        )
    });

    let exit_reason = loop {
        if start.elapsed() >= cfg.budget {
            break "time_budget";
        }
        if cfg.max_iters > 0 && iters_this_run >= cfg.max_iters {
            break "max_iters";
        }
        iters_this_run += 1;
        state.iter += 1;

        // Relative jitter + a small absolute floor (params span magnitudes
        // from ~0.003 to ~1.5; the floor lets near-zero params move and sign
        // flips stay possible). The RNG threads through the state as its raw
        // word so a resumed walk continues the same stream.
        let mut rng = SplitMix64::from_raw(state.rng_raw);
        let mut proposal = state.best_params;
        for p in proposal.iter_mut() {
            let scale = (f64::from(p.abs()) + 0.02) * state.sigma;
            *p += (scale * gauss(&mut rng)) as f32;
        }
        state.rng_raw = rng.into_raw();

        let mut accepted;
        let mut confirm_report = None;
        let mut anchor_action = None;
        if cfg.rotate {
            // Fresh disjoint block this iteration; incumbent and proposal race
            // on it head-to-head (paired CRN within the iteration, no reuse
            // across iterations — seed overfitting is structurally impossible).
            let block_seeds = seed_set_from(cfg.campaign.rotation_block(state.iter, block), block);
            let incumbent_score = objective(
                &state.best_params,
                &state.origin_params,
                cfg.width,
                cfg.depth,
                &block_seeds,
                cfg.format,
            );
            let proposal_score = objective(
                &proposal,
                &state.origin_params,
                cfg.width,
                cfg.depth,
                &block_seeds,
                cfg.format,
            );
            accepted = proposal_score > incumbent_score + cfg.accept_margin;
            eprintln!(
                "iter {} | screen {} {proposal_score:+.1} vs {incumbent_score:+.1} | sigma {:.3}",
                state.iter,
                if accepted { "PASS" } else { "reject" },
                state.sigma
            );

            // The confirmer: a screened proposal must SURVIVE-beat the current
            // incumbent in a sequential race before it may move the walk. H0
            // or an in-budget inconclusive demotes it — the v1/v2/v3 lesson is
            // that block means at this size pass noise; the racer does not.
            if accepted && cfg.confirm_matches > 0 {
                let report = race(
                    &proposal,
                    &state.best_params,
                    cfg,
                    cfg.campaign.confirm_base(state.iter, confirm_stride),
                    cfg.confirm_matches,
                    cfg.confirm_alpha,
                    deadline,
                );
                accepted = report.verdict == SprtVerdict::H1Accepted;
                eprintln!(
                    "iter {} | sprt {} | decisive {}-{} of {} pairs | LLR {:+.2} | {}",
                    state.iter,
                    match report.verdict {
                        SprtVerdict::H1Accepted => "CONFIRM",
                        SprtVerdict::H0Accepted => "DEMOTE (H0)",
                        SprtVerdict::Inconclusive => "DEMOTE (inconclusive)",
                    },
                    report.wins,
                    report.losses,
                    report.pairs,
                    report.llr,
                    if accepted { "adopting" } else { "discarding" },
                );
                confirm_report = Some(report);
            }
        } else {
            let score = objective(
                &proposal,
                &state.origin_params,
                cfg.width,
                cfg.depth,
                &cfg.train_seeds,
                cfg.format,
            );
            let best = fixed_best.get_or_insert(f64::NEG_INFINITY);
            accepted = score > *best;
            if accepted {
                *best = score;
                eprintln!(
                    "iter {} | ACCEPT {score:+.1} | sigma {:.3}",
                    state.iter, state.sigma
                );
            } else {
                eprintln!(
                    "iter {} | reject {score:+.1} (best {best:+.1}) | sigma {:.3}",
                    state.iter, state.sigma
                );
            }
        }

        if accepted {
            state.accepts += 1;
            state.best_params = proposal;
            // One-fifth-style adaptation: widen on success, narrow on failure.
            state.sigma = (state.sigma * 1.3).min(0.5);
            eprintln!(
                "iter {} | ACCEPT #{} | params {:?}",
                state.iter, state.accepts, state.best_params
            );

            // The anchor: every ANCHOR_EVERY confirmed accepts, the walk must
            // beat its last verified point end-to-end or return to it. This
            // bounds confirmation-alpha accumulation to one anchor window.
            if cfg.rotate
                && cfg.anchor_every > 0
                && state.accepts - state.accepts_at_last_anchor >= cfg.anchor_every
            {
                let event = state.anchor_events;
                state.anchor_events += 1;
                let report = race(
                    &state.best_params,
                    &state.anchored_params,
                    cfg,
                    cfg.campaign.anchor_base(event, anchor_stride),
                    cfg.anchor_matches,
                    0.05,
                    deadline,
                );
                let action = match report.verdict {
                    SprtVerdict::H1Accepted => {
                        state.anchored_params = state.best_params;
                        "re-anchored"
                    }
                    SprtVerdict::H0Accepted => {
                        state.best_params = state.anchored_params;
                        "ROLLED BACK"
                    }
                    SprtVerdict::Inconclusive => "kept old anchor (inconclusive)",
                };
                state.accepts_at_last_anchor = state.accepts;
                eprintln!(
                    "iter {} | anchor #{event} {action} | decisive {}-{} of {} pairs | LLR {:+.2}",
                    state.iter, report.wins, report.losses, report.pairs, report.llr
                );
                anchor_action = Some((report, action));
            }
        } else {
            state.sigma = (state.sigma * 0.95).max(0.02);
        }

        if let Some(l) = ledger.as_deref_mut() {
            let _ = l.append_outcome(&json!({
                "iter": state.iter,
                "accepted": accepted,
                "sigma": state.sigma,
                "confirm": confirm_report.as_ref().map(report_json),
                "anchor": anchor_action.as_ref().map(|(r, action)| {
                    let mut v = report_json(r);
                    v["action"] = json!(action);
                    v
                }),
                "params": accepted.then_some(state.best_params.as_slice()),
            }));
            let mut checkpoint = state.clone();
            checkpoint.consumed_secs += start.elapsed().as_secs();
            let _ = l.write_checkpoint(serde_json::to_value(&checkpoint).unwrap());
        }
    };

    state.consumed_secs += start.elapsed().as_secs();
    (state, exit_reason)
}

fn main() {
    let budget_secs = env_usize("TIME_BUDGET_SECS", 1800) as u64;
    let campaign_id = env_string("CAMPAIGN", "scratch");
    let resume_dir = env_string("RESUME", "");
    let max_iters = env_usize("MAX_ITERS", 0) as u32;
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let format = VersusFormat {
        max_plies: env_usize("MAX_PLIES", 240) as u32,
        rain_period: env_usize("RAIN_PERIOD", 8) as u32,
    };
    let sigma = env_f64("SIGMA", 0.15);
    let climb_seed = env_usize("CLIMB_SEED", 1) as u64;
    let campaign = Campaign::derive(&campaign_id);
    let cfg = ClimbConfig {
        width,
        depth,
        format,
        train_seeds: seed_set(env_usize("SEEDS", 24)),
        rotate: env_usize("ROTATE", 1) == 1,
        accept_margin: env_usize("ACCEPT_MARGIN", 25) as f64,
        confirm_matches: env_usize("CONFIRM_MATCHES", 800) as u32,
        confirm_alpha: env_f64("CONFIRM_ALPHA", 0.02),
        anchor_every: env_usize("ANCHOR_EVERY", 3) as u32,
        anchor_matches: env_usize("ANCHOR_MATCHES", 800) as u32,
        budget: Duration::from_secs(budget_secs),
        max_iters,
        campaign,
    };
    let val_count = env_usize("VAL_SEEDS", 32);
    let val_seeds = seed_set_from(cfg.campaign.validation(val_count), val_count);

    let state = if resume_dir.is_empty() {
        fresh_state(&cfg.campaign, sigma, climb_seed)
    } else {
        let checkpoint = RunLedger::read_checkpoint(Path::new(&resume_dir))
            .unwrap_or_else(|e| panic!("RESUME={resume_dir}: no readable checkpoint ({e})"));
        let state: ClimbState = serde_json::from_value(checkpoint)
            .unwrap_or_else(|e| panic!("RESUME={resume_dir}: checkpoint does not parse ({e})"));
        assert_eq!(
            state.campaign, cfg.campaign.id,
            "RESUME checkpoint belongs to campaign '{}' but CAMPAIGN is '{}'",
            state.campaign, cfg.campaign.id
        );
        state
    };

    // After every env read, so the spec captures the resolved config.
    let mut ledger = RunLedger::create(
        "versus_climb",
        json!({
            "campaign": { "id": cfg.campaign.id, "slot": cfg.campaign.slot },
            "resumed_from": (!resume_dir.is_empty()).then_some(&resume_dir),
            "bot": format!("beam(d{depth}, w{width}) cc2 attack_tuned+board_params"),
            "format": { "rain_period": format.rain_period, "max_plies": format.max_plies },
        }),
    )
    .expect("versus_climb: cannot create the run ledger");

    eprintln!(
        "Versus climb — campaign '{}' (slot {}) | beam(d{depth}, w{width}) vs attack_tuned | \
         {} train seeds x2, rain {}, {} plies | budget {budget_secs}s{}",
        cfg.campaign.id,
        cfg.campaign.slot,
        cfg.train_seeds.len(),
        format.rain_period,
        format.max_plies,
        if resume_dir.is_empty() {
            String::new()
        } else {
            format!(" | RESUMED from {resume_dir} at iter {}", state.iter)
        }
    );

    let (state, exit_reason) = run_climb(&cfg, state, Some(&mut ledger));
    eprintln!(
        "climb stopped ({exit_reason}): {} iters, {} accepts | resume with RESUME={}",
        state.iter,
        state.accepts,
        ledger.dir().display()
    );
    println!(
        "best_params {}",
        state
            .best_params
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Held-out validation: the honest verdict on DISJOINT campaign seeds.
    let val = |params: &BoardParams| -> (u32, u32, u32, u32, f64) {
        let cand = bot(params, width, depth);
        let incumbent = bot(&state.origin_params, width, depth);
        let fwd = evaluate_versus_format(&cand, &incumbent, &val_seeds, format);
        let rev = evaluate_versus_format(&incumbent, &cand, &val_seeds, format);
        let (mut dw, mut dl, mut margin) = (0u32, 0u32, 0.0f64);
        let (mut cap_w, mut cap_l) = (0u32, 0u32);
        for o in &fwd.outcomes {
            if o.b_topped && !o.a_topped {
                dw += 1;
            } else if o.a_topped && !o.b_topped {
                dl += 1;
            } else if o.attack_a > o.attack_b {
                cap_w += 1;
            } else if o.attack_b > o.attack_a {
                cap_l += 1;
            }
            margin += f64::from(o.attack_a) - f64::from(o.attack_b);
        }
        for o in &rev.outcomes {
            if o.a_topped && !o.b_topped {
                dw += 1;
            } else if o.b_topped && !o.a_topped {
                dl += 1;
            } else if o.attack_b > o.attack_a {
                cap_w += 1;
            } else if o.attack_a > o.attack_b {
                cap_l += 1;
            }
            margin += f64::from(o.attack_b) - f64::from(o.attack_a);
        }
        (
            dw,
            dl,
            cap_w,
            cap_l,
            margin / (2.0 * val_seeds.len() as f64),
        )
    };
    let (dw, dl, cw, cl, margin) = val(&state.best_params);
    eprintln!(
        "VALIDATION (held-out campaign seeds, {} x2): deaths won {dw} lost {dl} | \
         cap tiebreaks won {cw} lost {cl} | mean margin {margin:+.2}",
        val_seeds.len()
    );
    println!("val_death_wins {dw}");
    println!("val_death_losses {dl}");
    println!("val_mean_margin {margin:+.3}");

    let _ = ledger.write_summary(json!({
        "exit_reason": exit_reason,
        "iter": state.iter,
        "accepts": state.accepts,
        "anchor_events": state.anchor_events,
        "consumed_secs": state.consumed_secs,
        "best_params": state.best_params.as_slice(),
        "validation": {
            "death_wins": dw, "death_losses": dl,
            "cap_wins": cw, "cap_losses": cl,
            "mean_margin": margin,
        },
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(max_iters: u32) -> ClimbConfig {
        ClimbConfig {
            width: 4,
            depth: 1,
            format: VersusFormat {
                max_plies: 16,
                rain_period: 4,
            },
            train_seeds: seed_set(2),
            rotate: true,
            // Every screen passes: the test exercises the accept path, sigma
            // growth, and anchor bookkeeping deterministically.
            accept_margin: f64::NEG_INFINITY,
            confirm_matches: 0,
            confirm_alpha: 0.02,
            anchor_every: 2,
            // Zero-match anchor races resolve Inconclusive instantly — the
            // bookkeeping (events, counters) still advances and checkpoints.
            anchor_matches: 0,
            budget: Duration::from_secs(3600),
            max_iters,
            campaign: Campaign::derive("resume-bit-identity-test"),
        }
    }

    /// An interrupted walk continued from its checkpointed state must equal
    /// the uninterrupted walk bit-for-bit — params, sigma, RNG word, and
    /// anchor bookkeeping (wall-clock accounting excluded by construction).
    #[test]
    fn resume_reproduces_the_uninterrupted_walk() {
        let fresh = || fresh_state(&tiny_cfg(0).campaign, 0.15, 7);

        let (full, reason) = run_climb(&tiny_cfg(6), fresh(), None);
        assert_eq!(reason, "max_iters");

        let (half, _) = run_climb(&tiny_cfg(3), fresh(), None);
        let (resumed, _) = run_climb(&tiny_cfg(3), half, None);

        let scrub = |mut s: ClimbState| {
            s.consumed_secs = 0;
            serde_json::to_value(s).unwrap()
        };
        assert_eq!(scrub(full), scrub(resumed));
    }

    /// The state round-trips through its checkpoint encoding unchanged — the
    /// other half of the resume guarantee.
    #[test]
    fn checkpoint_encoding_round_trips() {
        let cfg = tiny_cfg(2);
        let (state, _) = run_climb(&cfg, fresh_state(&cfg.campaign, 0.15, 7), None);
        let encoded = serde_json::to_value(&state).unwrap();
        let decoded: ClimbState = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), encoded);
    }
}
