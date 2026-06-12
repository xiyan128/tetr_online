//! Versus weight climb: hill-climb CC2 board params against a fixed incumbent
//! under the rain format — the win-rate-climb instrument the roadmap calls for.
//!
//! Method: a (1+1)-ES with common random numbers. Each candidate plays the
//! incumbent on the SAME train seeds in BOTH orientations (arm swap), scored by
//! a dense paired objective: death verdict (±1000 — death is what decides
//! matches) plus the net-attack margin (the cap-game tiebreak). Proposals
//! jitter each param relative to its magnitude (plus a small absolute floor so
//! near-zero params can move); sigma adapts by an accept-rate rule. The climb
//! is fully reproducible from the spec's `climb_seed`, and self-bounded by the
//! runtime budget.
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
//! RUN RECORD v2 (same day, rotate, margin 25, sigma 0.15): rotation
//! killed seed memorization but the bar was far below the objective's noise —
//! with ±1000 death spikes a 48-match block mean has σ ≈ ±90, so 25 let lucky
//! proposals through (20 accepts / 56 iters), the accept-rate rule then GREW
//! sigma, and the parameters random-walked (tslot3 4.9 → 17.3); validation:
//! deaths 14-16, tiebreaks 8-57, margin −14.8. Gate caught it again; nothing
//! shipped. CALIBRATION LESSON: `accept_margin` must be ~2σ of the block mean
//! (~150-200 at these sizes), with sigma small (~0.08) so a rare false accept
//! cannot teleport the walk.
//!
//! RUN RECORD v3 (same day, margin 150, sigma 0.08): the calibrated
//! design behaved exactly as intended — 45 iters, ONE accept (+213.9 over a
//! +0.0 incumbent block), and the first POSITIVE held-out validation: deaths
//! 20-15, tiebreaks 32-28, margin +0.79. Not statistically significant
//! (20-15 of 35 decisive ⇒ p ≈ 0.25 one-sided), so NOT shipped — but the
//! candidate is a small, sane perturbation of attack_tuned worth a long SPRT
//! (pinned as the registry's `race-v3-candidate`).
//! SPRT EPILOGUE (same day): the v3 candidate was REJECTED —
//! H0 accepted in 270 s (266-269 of 544 decisive, LLR −2.99). The one accept
//! the calibrated climb produced in an hour was noise after all; the gate
//! chain (held-out validation → SPRT) worked end to end and nothing shipped.
//! Honest read: board-params-only jitter around attack_tuned has no cheap
//! survival gains at this match budget — further wins want a different lever
//! (re-priced garbage-aware weights, deeper search, or the NN round), not
//! more of this walk.
//!
//! THE CONFIRMER (same day, post-epilogue): the rotate path runs exactly
//! that design — a screened accept must additionally win an SPRT race
//! ([`crate::sprt`], proposal vs the walk's current incumbent, a fresh
//! per-iter region, capped by `confirm_matches`) before it moves the walk;
//! H0 or in-budget inconclusive DEMOTES it. Block means at these sizes pass
//! noise (v1/v2/v3 all proved it); the racer does not.
//!
//! THE ANCHOR (2026-06-11): per-accept confirmation bounds each STEP's
//! false-accept rate (`confirm_alpha`, default 0.02), but a long walk takes
//! many steps and the per-step α accumulates — eventually some confirmed
//! accept is noise, and the walk ratchets on an illusion. So every
//! `anchor_every` confirmed accepts, the walk must additionally beat its last
//! ANCHORED point (the last SPRT-verified composition of accepts) in a fresh
//! race: H1 re-anchors at the current params, H0 ROLLS the walk BACK to the
//! anchored point, inconclusive keeps the old anchor and retries after the
//! next window. Drift between anchors is therefore bounded by one anchor
//! window, and everything past the last anchor is always SPRT-verified
//! end-to-end — alpha accumulation buys noise for at most one window, never
//! the campaign.
//!
//! CAMPAIGN REGIONS + RESUME (2026-06-11): validation, rotation,
//! confirmation, and anchor seeds come from the campaign's private slab
//! ([`crate::seeds::Campaign`]) — fresh per campaign name, so repeated
//! campaigns stop iterating against one static validation region. Train (the
//! non-rotate path) stays at `regions::TRAIN`. Every run writes a
//! [`crate::ledger`] receipt, and the walk checkpoints its state into the
//! run directory each iteration; `resume <run-dir>` continues an interrupted walk
//! bit-identically from its checkpoint (state carries the RNG word, sigma,
//! iteration, params, and anchor bookkeeping — witnessed by the
//! `resume_reproduces_the_uninterrupted_walk` test), and refuses a registry
//! entry that drifted since the checkpoint. All pre-campaign trajectories
//! above (v1/v2/v3/confirmer) reproduce only at pre-move commits; their
//! verdicts stand — as do the env-var-era invocations this CLI replaced
//! (2026-06-12; the knobs map 1:1 onto [`Spec`] fields).

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::bots::{self, BotSpec, EvalSpec, SearchSpec};
use crate::commands::{BoardParams, Runtime};
use crate::ledger::RunDir;
use crate::rng::SplitMix64;
use crate::seeds::{Campaign, seed_set, seed_set_from};
use crate::sprt::{SprtConfig, SprtReport, SprtVerdict, sprt_race};
use crate::versus::{VersusFormat, evaluate_versus_format};

/// One standard-normal draw (Box-Muller over the deterministic SplitMix64).
fn gauss(rng: &mut SplitMix64) -> f64 {
    let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Everything that decides the walk — its campaign, opponent format, gates,
/// and RNG stream. Two runs of one spec are prefixes of the same trajectory.
#[derive(Clone, Debug, Serialize)]
pub struct Spec {
    /// The campaign whose private seed slab this climb draws from.
    pub campaign: String,
    /// The registered bot the walk starts from and mutates — must be a
    /// beam + CC2 bot; its weights are the campaign origin.
    pub subject: String,
    pub format: VersusFormat,
    /// Screen block size (rotate) / fixed train set size (non-rotate).
    pub screen_seeds: usize,
    /// Held-out validation seeds (campaign region).
    pub val_seeds: usize,
    /// Fresh disjoint screen block per iteration (the v2 regularization);
    /// `false` reproduces the v1 fixed-seed climb.
    pub rotate: bool,
    /// Screen bar on the paired block objective (calibrate to ~2σ).
    pub accept_margin: f64,
    /// Initial proposal jitter scale (adapts by the one-fifth rule).
    pub sigma: f64,
    /// The walk's RNG stream — the whole trajectory follows from it.
    pub climb_seed: u64,
    /// Per-accept confirmation race cap (0 disables the confirmer).
    pub confirm_matches: u32,
    pub confirm_alpha: f64,
    /// Anchor race every N confirmed accepts (0 disables anchoring).
    pub anchor_every: u32,
    pub anchor_matches: u32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            campaign: "scratch".to_string(),
            subject: "attack-tuned".to_string(),
            format: VersusFormat {
                max_plies: 240,
                rain_period: 8,
            },
            screen_seeds: 24,
            val_seeds: 32,
            rotate: true,
            accept_margin: 25.0,
            sigma: 0.15,
            climb_seed: 1,
            confirm_matches: 800,
            confirm_alpha: 0.02,
            anchor_every: 3,
            anchor_matches: 800,
        }
    }
}

/// The resolved climb subject: the search shape the walk keeps and the CC2
/// weights whose board params it mutates.
struct Subject {
    base: tetr_core::ai::Cc2Weights,
    width: usize,
    depth: u8,
}

fn resolve_subject(name: &str) -> Subject {
    let bot = bots::find(name)
        .unwrap_or_else(|| panic!("climb subject {name:?} is not a registered bot"));
    let SearchSpec::Beam { width, depth } = bot.spec.search else {
        panic!("climb subject {name:?} must be a beam bot");
    };
    let EvalSpec::Cc2(base) = bot.spec.eval else {
        panic!("climb subject {name:?} must use the CC2 evaluator");
    };
    Subject { base, width, depth }
}

/// Default wall-clock budget (`--budget-secs` overrides).
const DEFAULT_BUDGET_SECS: u64 = 1800;

/// The walk's complete resumable state — everything the next iteration reads.
/// A checkpointed state plus the same spec reproduces the uninterrupted
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
    /// each invocation is freshly bounded by its own budget).
    consumed_secs: u64,
}

fn fresh_state(campaign: &Campaign, subject: &Subject, sigma: f64, climb_seed: u64) -> ClimbState {
    let origin = subject.base.board_params();
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

fn bot(
    params: &BoardParams,
    subject: &Subject,
) -> impl Fn(u64) -> Box<dyn tetr_core::player::PlayerController> + Send + Sync + 'static {
    BotSpec::beam(subject.width, subject.depth)
        .cc2(subject.base.with_board_params(params))
        .factory()
}

/// The paired objective of `candidate` vs the incumbent over `seeds`, both
/// orientations: mean over matches of `1000·(they died − we died) + (our net
/// attack − theirs)`. Higher is better; the death term dominates by design
/// (death decides matches; margin orders the rest).
fn objective(
    params: &BoardParams,
    incumbent_params: &BoardParams,
    subject: &Subject,
    seeds: &[u64],
    format: VersusFormat,
) -> f64 {
    let cand = bot(params, subject);
    let incumbent = bot(incumbent_params, subject);

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

fn race(
    cand_params: &BoardParams,
    incumbent_params: &BoardParams,
    subject: &Subject,
    format: VersusFormat,
    config: SprtConfig,
) -> SprtReport {
    sprt_race(
        &bot(cand_params, subject),
        &bot(incumbent_params, subject),
        format,
        config,
    )
}

/// Run (or continue) the walk until the budget or `max_iters` ends this
/// invocation. Returns the final state and the exit reason.
fn run_climb(
    spec: &Spec,
    subject: &Subject,
    campaign: &Campaign,
    budget: std::time::Duration,
    max_iters: u32,
    mut state: ClimbState,
    checkpoint: Option<&RunDir>,
) -> (ClimbState, &'static str) {
    let start = Instant::now();
    let deadline = start + budget;
    let block = spec.screen_seeds;
    let train_seeds = seed_set(spec.screen_seeds);
    // Stride ≥ the race's worst-case seed consumption (M matches use at most
    // M/2 indices; M covers it with slack), and never below the historical
    // 4096 so overridden caps cannot make successive races overlap.
    let confirm_stride = 4096usize.max(spec.confirm_matches as usize);
    let anchor_stride = 4096usize.max(spec.anchor_matches as usize);
    let mut iters_this_run = 0u32;
    // The non-rotate path's running best — re-derived deterministically on
    // resume (same params, same fixed train seeds).
    let mut fixed_best = (!spec.rotate).then(|| {
        objective(
            &state.best_params,
            &state.origin_params,
            subject,
            &train_seeds,
            spec.format,
        )
    });

    let exit_reason = loop {
        if start.elapsed() >= budget {
            break "time_budget";
        }
        if max_iters > 0 && iters_this_run >= max_iters {
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
        if spec.rotate {
            // Fresh disjoint block this iteration; incumbent and proposal race
            // on it head-to-head (paired CRN within the iteration, no reuse
            // across iterations — seed overfitting is structurally impossible).
            let block_seeds = seed_set_from(campaign.rotation_block(state.iter, block), block);
            let incumbent_score = objective(
                &state.best_params,
                &state.origin_params,
                subject,
                &block_seeds,
                spec.format,
            );
            let proposal_score = objective(
                &proposal,
                &state.origin_params,
                subject,
                &block_seeds,
                spec.format,
            );
            accepted = proposal_score > incumbent_score + spec.accept_margin;
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
            if accepted && spec.confirm_matches > 0 {
                let report = race(
                    &proposal,
                    &state.best_params,
                    subject,
                    spec.format,
                    SprtConfig {
                        alpha: spec.confirm_alpha,
                        seed_base: campaign.confirm_base(state.iter, confirm_stride),
                        max_matches: spec.confirm_matches,
                        deadline: Some(deadline),
                        ..SprtConfig::default()
                    },
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
            }
        } else {
            let score = objective(
                &proposal,
                &state.origin_params,
                subject,
                &train_seeds,
                spec.format,
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

            // The anchor: every `anchor_every` confirmed accepts, the walk
            // must beat its last verified point end-to-end or return to it.
            // This bounds confirmation-alpha accumulation to one window.
            if spec.rotate
                && spec.anchor_every > 0
                && state.accepts - state.accepts_at_last_anchor >= spec.anchor_every
            {
                let event = state.anchor_events;
                state.anchor_events += 1;
                let report = race(
                    &state.best_params,
                    &state.anchored_params,
                    subject,
                    spec.format,
                    SprtConfig {
                        alpha: 0.05,
                        seed_base: campaign.anchor_base(event, anchor_stride),
                        max_matches: spec.anchor_matches,
                        deadline: Some(deadline),
                        ..SprtConfig::default()
                    },
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
            }
        } else {
            state.sigma = (state.sigma * 0.95).max(0.02);
        }

        if let Some(dir) = checkpoint {
            let mut snapshot = state.clone();
            snapshot.consumed_secs += start.elapsed().as_secs();
            let _ = dir.write_checkpoint(serde_json::to_value(&snapshot).unwrap());
        }
    };

    state.consumed_secs += start.elapsed().as_secs();
    (state, exit_reason)
}

pub fn run(spec: &Spec, rt: &Runtime, run_dir: &RunDir) -> std::io::Result<()> {
    let subject = resolve_subject(&spec.subject);
    let campaign = Campaign::derive(&spec.campaign);
    let state = fresh_state(&campaign, &subject, spec.sigma, spec.climb_seed);
    drive(spec, &subject, rt, campaign, state, run_dir)
}

/// Continue an interrupted walk from `prior` (a run directory with a
/// checkpoint). The registry-drift check happened in `main`; the campaign
/// assert below is the in-checkpoint belt to that suspender.
pub fn resume(spec: &Spec, rt: &Runtime, prior: &Path, run_dir: &RunDir) -> std::io::Result<()> {
    let subject = resolve_subject(&spec.subject);
    let campaign = Campaign::derive(&spec.campaign);
    let checkpoint = RunDir::read_checkpoint(prior)?;
    let state: ClimbState = serde_json::from_value(checkpoint).map_err(std::io::Error::other)?;
    assert_eq!(
        state.campaign, campaign.id,
        "checkpoint belongs to campaign '{}' but the spec says '{}'",
        state.campaign, campaign.id
    );
    eprintln!("RESUMED from {} at iter {}", prior.display(), state.iter);
    drive(spec, &subject, rt, campaign, state, run_dir)
}

fn drive(
    spec: &Spec,
    subject: &Subject,
    rt: &Runtime,
    campaign: Campaign,
    state: ClimbState,
    run_dir: &RunDir,
) -> std::io::Result<()> {
    let budget = rt.budget(DEFAULT_BUDGET_SECS);
    eprintln!(
        "Versus climb — campaign '{}' (slot {}) | beam(d{}, w{}) vs attack_tuned | \
         {} train seeds x2, rain {}, {} plies | budget {}s",
        campaign.id,
        campaign.slot,
        subject.depth,
        subject.width,
        spec.screen_seeds,
        spec.format.rain_period,
        spec.format.max_plies,
        budget.as_secs(),
    );

    let (state, exit_reason) = run_climb(
        spec,
        subject,
        &campaign,
        budget,
        rt.max_iters,
        state,
        Some(run_dir),
    );
    eprintln!(
        "climb stopped ({exit_reason}): {} iters, {} accepts | continue with `resume {}`",
        state.iter,
        state.accepts,
        run_dir.dir().display()
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
    let val_seeds = seed_set_from(campaign.validation(spec.val_seeds), spec.val_seeds);
    let cand = bot(&state.best_params, subject);
    let incumbent = bot(&state.origin_params, subject);
    let fwd = evaluate_versus_format(&cand, &incumbent, &val_seeds, spec.format);
    let rev = evaluate_versus_format(&incumbent, &cand, &val_seeds, spec.format);
    let (mut dw, mut dl, mut margin) = (0u32, 0u32, 0.0f64);
    let (mut cw, mut cl) = (0u32, 0u32);
    for o in &fwd.outcomes {
        if o.b_topped && !o.a_topped {
            dw += 1;
        } else if o.a_topped && !o.b_topped {
            dl += 1;
        } else if o.attack_a > o.attack_b {
            cw += 1;
        } else if o.attack_b > o.attack_a {
            cl += 1;
        }
        margin += f64::from(o.attack_a) - f64::from(o.attack_b);
    }
    for o in &rev.outcomes {
        if o.a_topped && !o.b_topped {
            dw += 1;
        } else if o.b_topped && !o.a_topped {
            dl += 1;
        } else if o.attack_b > o.attack_a {
            cw += 1;
        } else if o.attack_a > o.attack_b {
            cl += 1;
        }
        margin += f64::from(o.attack_b) - f64::from(o.attack_a);
    }
    let margin = margin / (2.0 * val_seeds.len() as f64);
    eprintln!(
        "VALIDATION (held-out campaign seeds, {} x2): deaths won {dw} lost {dl} | \
         cap tiebreaks won {cw} lost {cl} | mean margin {margin:+.2}",
        val_seeds.len()
    );
    println!("val_death_wins {dw}");
    println!("val_death_losses {dl}");
    println!("val_mean_margin {margin:+.3}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tiny_spec() -> Spec {
        Spec {
            campaign: "resume-bit-identity-test".to_string(),
            subject: "attack-tuned-tiny".to_string(),
            format: VersusFormat {
                max_plies: 16,
                rain_period: 4,
            },
            screen_seeds: 2,
            val_seeds: 2,
            rotate: true,
            // Every screen passes: the test exercises the accept path, sigma
            // growth, and anchor bookkeeping deterministically.
            accept_margin: f64::NEG_INFINITY,
            sigma: 0.15,
            climb_seed: 7,
            confirm_matches: 0,
            confirm_alpha: 0.02,
            anchor_every: 2,
            // Zero-match anchor races resolve Inconclusive instantly — the
            // bookkeeping (events, counters) still advances and checkpoints.
            anchor_matches: 0,
        }
    }

    fn go(spec: &Spec, max_iters: u32, state: ClimbState) -> (ClimbState, &'static str) {
        let campaign = Campaign::derive(&spec.campaign);
        run_climb(
            spec,
            &resolve_subject(&spec.subject),
            &campaign,
            Duration::from_secs(3600),
            max_iters,
            state,
            None,
        )
    }

    /// An interrupted walk continued from its checkpointed state must equal
    /// the uninterrupted walk bit-for-bit — params, sigma, RNG word, and
    /// anchor bookkeeping (wall-clock accounting excluded by construction).
    #[test]
    fn resume_reproduces_the_uninterrupted_walk() {
        let spec = tiny_spec();
        let fresh = || {
            fresh_state(
                &Campaign::derive(&spec.campaign),
                &resolve_subject(&spec.subject),
                spec.sigma,
                spec.climb_seed,
            )
        };

        let (full, reason) = go(&spec, 6, fresh());
        assert_eq!(reason, "max_iters");

        let (half, _) = go(&spec, 3, fresh());
        let (resumed, _) = go(&spec, 3, half);

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
        let spec = tiny_spec();
        let (state, _) = go(
            &spec,
            2,
            fresh_state(
                &Campaign::derive(&spec.campaign),
                &resolve_subject(&spec.subject),
                spec.sigma,
                spec.climb_seed,
            ),
        );
        let encoded = serde_json::to_value(&state).unwrap();
        let decoded: ClimbState = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), encoded);
    }
}
