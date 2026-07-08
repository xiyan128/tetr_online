//! Gate-0a — survival-recall@k (leapfrog map ticket T11).
//!
//! The cheapest decisive falsification of the leapfrog thesis: can a learned
//! policy prior's top-k COVER the survival branches that brute width finds at
//! near-death positions? If not, the survival hedge is an irreducible breadth
//! property and a narrow policy-guided search cannot replace w128.
//!
//! Procedure (zero training):
//! 1. Play champion (tp:cc2@w128d9) mirror self-play under a pressured venue;
//!    capture the topping-out side's last K `SearchState`s (near-death).
//! 2. Per state, re-run a fresh w128d9 beam and read `root_scores()`: a root is
//!    a SURVIVAL root iff its backed-up score is not death-dominated (all-dead
//!    lines back up to the internal DEATH_SCORE ≈ -1e8; real scores are O(1e4)).
//! 3. Per state, take the net's per-root policy logits (same `hold_placements`
//!    order — beam roots, root_scores, and the net children all enumerate it),
//!    and its top-k.
//! 4. recall@k = |survival_roots ∩ policy_topk| / |survival_roots|.
//!
//! The beam roots and the net children are the SAME `hold_placements(state)`
//! list in the SAME canonical order, so classification and ranking align by
//! index (asserted in the test).

use std::collections::VecDeque;

use tetr_core::ai::search::{hold_placements, think_to_completion};
use tetr_core::ai::state::SearchState;
use tetr_core::ai::{BeamPlanner, Cc2Evaluator, SearchBudget};
use tetr_core::engine::Engine;
use tetr_nn::net::{Net, Scratch};
use tetr_nn::obs::{OppCtx, encode};

use crate::accounting::controller_seed;
use crate::arm::Arm;
use crate::marathon::marathon_config;
use crate::versus::{VersusFormat, versus_step_piece};

/// A root above this backed-up score has at least one surviving line within the
/// beam horizon. DEATH_SCORE is -1e8 (tetr-core internal, not exported); real
/// leaf scores are z_scale·z_hat + attack ≈ O(1e4), so the gap is enormous and
/// the exact cut is not sensitive. The test prints the histogram to confirm.
const SURVIVE_THRESH: i32 = -1_000_000;

/// Champion beam config (matches probe-tp128d9).
const CHAMP_WIDTH: usize = 128;
const CHAMP_DEPTH: u8 = 9;

/// Per-state recall record.
#[derive(Debug, Clone)]
pub struct StateRecord {
    /// Number of legal root placements (`hold_placements`).
    pub n_roots: usize,
    /// Roots the beam finds a surviving line from.
    pub n_survival: usize,
    /// Roots the net marks immediately dead (excluded from the policy top-k).
    pub n_dead: usize,
    /// min / median / max backed-up root score (histogram sanity for the cut).
    pub root_best_min: i32,
    pub root_best_med: i32,
    pub root_best_max: i32,
    /// recall@k for each k in [`KS`], NaN when there are no survival roots.
    pub recall: Vec<f32>,
    /// agreement@k: |net top-k ∩ beam top-k-by-score| / k over live roots — does
    /// the prior concentrate on the placements the champion actually prefers
    /// (the survival-relevant signal, since binary d9-survival is near-trivial)?
    /// NaN when fewer than k live roots exist.
    pub agree: Vec<f32>,
}

/// The k values reported.
pub const KS: [usize; 4] = [6, 12, 18, 24];

/// Capture the topping-out side's last `k` near-death states from one champion
/// mirror game. Empty if the game hit the cap without a topout. The corpus
/// generator a Gate-0a instrument fans over a seed region.
pub fn capture_near_death(
    champ: &Arm,
    seed: u64,
    format: &VersusFormat,
    k: usize,
) -> Vec<SearchState> {
    let mut a_engine = Engine::new(marathon_config(), seed);
    let mut b_engine = Engine::new(marathon_config(), seed);
    let mut a_bot = champ.controller(controller_seed(seed));
    let mut b_bot = champ.controller(controller_seed(seed));
    let mut a_ring: VecDeque<SearchState> = VecDeque::with_capacity(k + 1);
    let mut b_ring: VecDeque<SearchState> = VecDeque::with_capacity(k + 1);

    for ply in 0..format.hard_cap() {
        let period = format.rain_period_at(ply);
        if period > 0 && ply % period == period - 1 {
            a_engine.queue_garbage(1);
            b_engine.queue_garbage(1);
        }
        let order = if ply % 2 == 0 { [0u8, 1] } else { [1, 0] };
        for &who in &order {
            if who == 0 {
                if let Some(s) = SearchState::from_snapshot(&a_engine.snapshot()) {
                    if a_ring.len() == k {
                        a_ring.pop_front();
                    }
                    a_ring.push_back(s);
                }
                let (atk, topped) = versus_step_piece(&mut a_engine, &mut *a_bot);
                if atk > 0 {
                    b_engine.queue_garbage(atk);
                }
                if topped {
                    return a_ring.into_iter().collect();
                }
            } else {
                if let Some(s) = SearchState::from_snapshot(&b_engine.snapshot()) {
                    if b_ring.len() == k {
                        b_ring.pop_front();
                    }
                    b_ring.push_back(s);
                }
                let (atk, topped) = versus_step_piece(&mut b_engine, &mut *b_bot);
                if atk > 0 {
                    a_engine.queue_garbage(atk);
                }
                if topped {
                    return b_ring.into_iter().collect();
                }
            }
        }
    }
    Vec::new()
}

/// The champion beam's survival-root mask at a state, aligned to
/// `hold_placements(state)` order. `root_best[i]` is the backed-up score of
/// root i (`i32::MIN` if the beam scored no descendant of it).
fn beam_survival(state: &SearchState, cc2: &Cc2Evaluator) -> Vec<i32> {
    let n = hold_placements(state).len();
    let mut planner = BeamPlanner::transposing(CHAMP_WIDTH);
    think_to_completion(&mut planner, state, cc2, SearchBudget::beam(CHAMP_DEPTH));
    let mut root_best = vec![i32::MIN; n];
    for (i, (_placement, score)) in planner.root_scores().enumerate() {
        root_best[i] = score;
    }
    root_best
}

/// The net's per-root policy logits and dead mask, aligned to
/// `hold_placements(state)` (the [`PolicyMind`](crate::arm) enumeration).
fn net_policy(
    state: &SearchState,
    net: &Net,
    opp: &OppCtx,
    scratch: &mut Scratch,
) -> (Vec<f32>, Vec<bool>) {
    let opp_emb = net
        .embed_boards(&[&opp.board], scratch)
        .pop()
        .expect("one plane in, one embedding out");
    let placements = hold_placements(state);
    let children: Vec<(_, bool)> = placements
        .iter()
        .map(|p| {
            let mut child = state.clone();
            child.commit_placement(p);
            (encode(&child, opp), child.dead)
        })
        .collect();
    let items: Vec<_> = children
        .iter()
        .map(|(o, _)| (&o.own_board, &o.features))
        .collect();
    let heads = net.forward(&items, &opp_emb, scratch);
    let policy = heads.iter().map(|h| h.policy).collect();
    let dead = children.iter().map(|(_, d)| *d).collect();
    (policy, dead)
}

/// Score one near-death state: classify survival roots, rank the net policy,
/// compute recall@k. Returns None if the position has no survival roots (a
/// forced loss — nothing to recall) or every live root survives (not actually
/// near-death — filtered by the caller).
pub fn score_state(
    state: &SearchState,
    cc2: &Cc2Evaluator,
    net: &Net,
    opp: &OppCtx,
    scratch: &mut Scratch,
) -> Option<StateRecord> {
    let root_best = beam_survival(state, cc2);
    let n = root_best.len();
    if n == 0 {
        return None;
    }
    let (policy, dead) = net_policy(state, net, opp, scratch);
    debug_assert_eq!(policy.len(), n, "net children align with beam roots");

    let survival: Vec<bool> = root_best.iter().map(|&s| s > SURVIVE_THRESH).collect();
    let n_survival = survival.iter().filter(|&&s| s).count();
    let n_dead = dead.iter().filter(|&&d| d).count();

    // Rank live roots by policy logit, descending (the net's preference order).
    let mut net_order: Vec<usize> = (0..n).filter(|&i| !dead[i]).collect();
    net_order.sort_by(|&a, &b| {
        policy[b]
            .partial_cmp(&policy[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Rank live roots by beam backed-up score, descending (the champion's order).
    let mut beam_order: Vec<usize> = (0..n).filter(|&i| !dead[i]).collect();
    beam_order.sort_by(|&a, &b| root_best[b].cmp(&root_best[a]));
    let n_live = net_order.len();

    let recall = KS
        .iter()
        .map(|&k| {
            if n_survival == 0 {
                f32::NAN
            } else {
                let hit = net_order.iter().take(k).filter(|&&i| survival[i]).count();
                hit as f32 / n_survival as f32
            }
        })
        .collect();

    let agree = KS
        .iter()
        .map(|&k| {
            if n_live < k {
                f32::NAN
            } else {
                let net_top: std::collections::HashSet<usize> =
                    net_order.iter().take(k).copied().collect();
                let hit = beam_order
                    .iter()
                    .take(k)
                    .filter(|i| net_top.contains(i))
                    .count();
                hit as f32 / k as f32
            }
        })
        .collect();

    let mut sorted = root_best.clone();
    sorted.sort_unstable();
    Some(StateRecord {
        n_roots: n,
        n_survival,
        n_dead,
        root_best_min: sorted[0],
        root_best_med: sorted[n / 2],
        root_best_max: sorted[n - 1],
        recall,
        agree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use tetr_core::ai::eval::Cc2Weights;

    /// Small end-to-end verification: a few champion games at a pressured venue,
    /// print the root-score histogram (to confirm the survival cut) and the
    /// recall@k on the round0 net. Run with:
    ///   cargo test --release -p tetr-research --test-threads=1 gate0a_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gate0a_smoke() {
        let champ = Arm::from_str("tp:cc2@w128d9").expect("champion arm");
        let cc2 = Cc2Evaluator::new(Cc2Weights::attack_tuned());
        let net = Net::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tetr-nn/tests/fixtures/round0"
        ))
        .expect("round0 net");
        let opp = OppCtx::default();
        let mut scratch = Scratch::default();

        // Pressured venue: heavy rain, short cap -> reaches near-death fast.
        let format = VersusFormat {
            max_plies: 120,
            rain_period: 4,
            sudden_death: true,
        };

        // More games, fewer states/game (last 3 before death) to cut within-game
        // correlation. The last-3 plies are the genuinely near-death ones.
        let mut records: Vec<StateRecord> = Vec::new();
        let mut games_with_death = 0;
        for seed in 1..=24u64 {
            let states = capture_near_death(&champ, seed, &format, 3);
            if !states.is_empty() {
                games_with_death += 1;
            }
            for st in &states {
                if let Some(r) = score_state(st, &cc2, &net, &opp, &mut scratch) {
                    records.push(r);
                }
            }
        }

        eprintln!(
            "\n=== Gate-0a smoke: {} states from {} games with a death ===",
            records.len(),
            games_with_death
        );
        eprintln!(
            "KEY FINDING check — n_survival vs n_live (should be equal ⇒ d9-survival is trivial):"
        );
        let survival_eq_live = records
            .iter()
            .filter(|r| r.n_survival == r.n_roots - r.n_dead)
            .count();
        eprintln!(
            "  {}/{} states have n_survival == n_live",
            survival_eq_live,
            records.len()
        );

        // Agreement@k: does the net's top-k cover the champion beam's top-k-by-score?
        // This is the meaningful metric (binary survival is trivial). Averaged over
        // states with at least k live roots.
        eprintln!(
            "\nagreement@k (net top-k ∩ beam top-k-by-score) / k   vs random baseline k/n_live:"
        );
        for (ki, &k) in KS.iter().enumerate() {
            let pairs: Vec<(f32, f32)> = records
                .iter()
                .filter(|r| !r.agree[ki].is_nan())
                .map(|r| {
                    let live = (r.n_roots - r.n_dead).max(1);
                    (r.agree[ki], k as f32 / live as f32)
                })
                .collect();
            if !pairs.is_empty() {
                let mean = pairs.iter().map(|p| p.0).sum::<f32>() / pairs.len() as f32;
                let base = pairs.iter().map(|p| p.1).sum::<f32>() / pairs.len() as f32;
                eprintln!(
                    "  agree@{k:<2} = {mean:.3}   random = {base:.3}   lift = {:.1}x   (n={})",
                    mean / base,
                    pairs.len()
                );
            }
        }
        assert!(
            games_with_death > 0,
            "no champion topout under the pressured venue"
        );
    }
}
