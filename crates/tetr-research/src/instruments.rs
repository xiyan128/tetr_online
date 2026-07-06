//! The versus instruments: `duel` and `gate`, over any two [`Arm`]s.
//!
//! Both play **CRN seed pairs** — each seed is played twice with the arms
//! swapped, so piece luck and first-mover structure cancel — under the
//! sudden-death venue (outcomes from death, never an attack tiebreak).
//!
//! - **duel**: a fixed number of pairs; reports W-L-D for arm A with the
//!   end-reason split. `duel --a beam:M@w8d5 --b policy:M` *is* the G_π
//!   probe; any candidate-vs-incumbent strength race is the same command.
//! - **gate**: the promotion instrument — a trinomial pair-GSPRT whose
//!   verdict **latches at the first boundary crossing**. Pairs still in
//!   flight when the verdict lands are reported (`post_decision_pairs`) but
//!   never enter the decision statistics — the sequential test's error
//!   bounds survive parallelism. (The reference campaign recomputed the
//!   verdict from augmented statistics; ~36% of marginal crossings flipped
//!   to Inconclusive on 16 cores. That failure mode is structural here,
//!   not patched.)
//!
//! Seeds are explicit everywhere (`--seeds BASE` + count): no instrument has
//! a hardcoded region, and the receipt records exactly what ran.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::json;

use crate::arm::Arm;
use crate::sprt::{SprtState, SprtVerdict};
use crate::versus::{EndReason, VersusFormat, VersusOutcome, VersusResult, play_versus_format};

/// The sudden-death venue both instruments play under.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Venue {
    pub max_plies: u32,
    pub rain_period: u32,
}

impl Default for Venue {
    /// The calibrated venue: rain 8, cap 240, sudden death (measured ~80%
    /// decisive pre-escalation, 100% decisive overall, 0 true caps in 2,200+
    /// reference games).
    fn default() -> Self {
        Self {
            max_plies: 240,
            rain_period: 8,
        }
    }
}

impl Venue {
    fn format(&self) -> VersusFormat {
        VersusFormat {
            max_plies: self.max_plies,
            rain_period: self.rain_period,
            sudden_death: true,
        }
    }
}

/// One CRN pair: the same seed with the arms swapped.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PairOutcome {
    pub seed: u64,
    /// A as first mover.
    pub forward: VersusOutcome,
    /// B as first mover (result still reported from A's perspective by
    /// [`PairOutcome::a_wins_losses`]).
    pub reverse: VersusOutcome,
}

impl PairOutcome {
    /// `(wins, losses)` for arm A across the pair (0..=2 each).
    pub fn a_wins_losses(&self) -> (u32, u32) {
        let mut w = 0;
        let mut l = 0;
        match self.forward.result {
            VersusResult::AWins => w += 1,
            VersusResult::BWins => l += 1,
            VersusResult::Draw => {}
        }
        match self.reverse.result {
            VersusResult::AWins => l += 1, // reverse: A is seat B
            VersusResult::BWins => w += 1,
            VersusResult::Draw => {}
        }
        (w, l)
    }
}

fn play_pair(a: &Arm, b: &Arm, seed: u64, format: VersusFormat) -> PairOutcome {
    PairOutcome {
        seed,
        forward: play_versus_format(&a.factory(), &b.factory(), seed, format),
        reverse: play_versus_format(&b.factory(), &a.factory(), seed, format),
    }
}

fn end_reason_hist(outcomes: impl Iterator<Item = EndReason>) -> [u32; 3] {
    let mut hist = [0u32; 3];
    for r in outcomes {
        hist[match r {
            EndReason::Topout => 0,
            EndReason::Escalation => 1,
            EndReason::TrueCap => 2,
        }] += 1;
    }
    hist
}

/// `duel`: `pairs` CRN pairs from `seed_base`, in parallel, budget-bounded.
/// Under truncation the pairs played are a true PREFIX of the seed range
/// (seeds `[seed_base, seed_base + pairs_played)`), so `pairs_played` fully
/// determines which seeds ran — reproducible, never a scattered subset.
pub fn duel(
    a: &Arm,
    b: &Arm,
    venue: Venue,
    seed_base: u64,
    pairs: usize,
    budget: Duration,
) -> serde_json::Value {
    let t0 = Instant::now();
    let format = venue.format();
    // Play seeds in order, one parallel chunk at a time, and stop at a chunk
    // boundary once the budget is spent — so the played set is a contiguous
    // prefix, not whichever pairs rayon's work-stealing happened to finish.
    let chunk = rayon::current_num_threads().max(1);
    let mut outcomes: Vec<PairOutcome> = Vec::with_capacity(pairs);
    for start in (0..pairs).step_by(chunk) {
        if t0.elapsed() >= budget {
            break;
        }
        let end = (start + chunk).min(pairs);
        let batch: Vec<PairOutcome> = (start..end)
            .into_par_iter()
            .map(|i| play_pair(a, b, seed_base + i as u64, format))
            .collect();
        outcomes.extend(batch);
    }

    let (mut wins, mut losses, mut draws) = (0u32, 0u32, 0u32);
    for p in &outcomes {
        let (w, l) = p.a_wins_losses();
        wins += w;
        losses += l;
        draws += 2 - w - l;
    }
    let hist = end_reason_hist(
        outcomes
            .iter()
            .flat_map(|p| [p.forward.end_reason, p.reverse.end_reason]),
    );
    let games = 2 * outcomes.len() as u32;
    eprintln!(
        "duel | A {wins}-{losses}-{draws} over {games} games | end {hist:?} | {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    json!({
        "a": a.to_string(), "b": b.to_string(),
        "wins_a": wins, "losses_a": losses, "draws": draws,
        // A's share of DECISIVE games (a draw is a tie: double death or cap).
        "a_win_share_decisive": f64::from(wins) / f64::from((wins + losses).max(1)),
        "end_reason_topout_escalation_truecap": hist,
        "pairs_asked": pairs, "pairs_played": outcomes.len(),
        "seed_base": seed_base,
        "wall_secs": t0.elapsed().as_secs_f64(),
    })
}

/// The gate's shared decision state: the sequential test plus the latch.
struct GateState {
    sprt: SprtState,
    /// Set exactly once, at the first boundary crossing. After this, no pair
    /// enters `sprt` — the sequential test is OVER; stragglers only count
    /// themselves in `post_decision_pairs`.
    decided: Option<SprtVerdict>,
    post_decision_pairs: u32,
}

/// `gate`: latched trinomial pair-GSPRT of `a` (candidate) vs `b` (incumbent).
#[allow(clippy::too_many_arguments)]
pub fn gate(
    a: &Arm,
    b: &Arm,
    venue: Venue,
    seed_base: u64,
    max_pairs: usize,
    p1: f64,
    min_pairs: u32,
    budget: Duration,
) -> serde_json::Value {
    let t0 = Instant::now();
    let format = venue.format();
    let state = Mutex::new(GateState {
        sprt: SprtState::new(p1, 0.05, 0.05, min_pairs),
        decided: None,
        post_decision_pairs: 0,
    });
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(max_pairs.max(1));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    // Stop pulling once decided or out of work/time.
                    if state.lock().expect("gate state").decided.is_some() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= max_pairs || t0.elapsed() > budget {
                        break;
                    }
                    let pair = play_pair(a, b, seed_base + i as u64, format);
                    let (w, l) = pair.a_wins_losses();

                    let mut st = state.lock().expect("gate state");
                    if st.decided.is_some() {
                        // The test ended while this pair was in flight: it is
                        // evidence collected after the decision — report it,
                        // never let it move the verdict.
                        st.post_decision_pairs += 1;
                        continue;
                    }
                    st.sprt.record_pair(w, l);
                    if let Some(v) = st.sprt.verdict() {
                        st.decided = Some(v); // the latch
                    }
                }
            });
        }
    });

    let st = state.into_inner().expect("gate state");
    let verdict = st.decided.unwrap_or(SprtVerdict::Inconclusive);
    let truncated = st.decided.is_none() && t0.elapsed() > budget;
    eprintln!(
        "gate | {verdict:?} after {} pairs (llr {:.3}) | {} in-flight excluded | {:.0}s",
        st.sprt.pairs(),
        st.sprt.llr(),
        st.post_decision_pairs,
        t0.elapsed().as_secs_f64()
    );
    json!({
        "a": a.to_string(), "b": b.to_string(),
        "verdict": format!("{verdict:?}"),
        "pairs": st.sprt.pairs(),
        "pair_counts": st.sprt.pair_counts(),
        "llr": st.sprt.llr(),
        "post_decision_pairs": st.post_decision_pairs,
        "budget_truncated": truncated,
        "p1": p1, "min_pairs": min_pairs,
        "seed_base": seed_base,
        "wall_secs": t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The latch under adversarial concurrency: many threads race pairs whose
    /// results would (without the latch) drag the LLR back inside the bounds.
    /// We drive the state machine directly — the property is about the
    /// bookkeeping, not the games.
    #[test]
    fn the_verdict_latches_at_first_crossing() {
        let mut st = GateState {
            sprt: SprtState::new(0.55, 0.05, 0.05, 4),
            decided: None,
            post_decision_pairs: 0,
        };
        // Feed candidate sweeps until H1 crosses…
        let mut crossing_pairs = 0;
        while st.decided.is_none() {
            st.sprt.record_pair(2, 0);
            crossing_pairs += 1;
            if let Some(v) = st.sprt.verdict() {
                st.decided = Some(v);
            }
            assert!(crossing_pairs < 10_000, "must cross eventually");
        }
        assert_eq!(st.decided, Some(SprtVerdict::H1Accepted));
        let (pairs_at_decision, llr_at_decision) = (st.sprt.pairs(), st.sprt.llr());

        // …then a storm of in-flight losses lands. With the latch they are
        // counted separately and the decision statistics do not move.
        for _ in 0..64 {
            if st.decided.is_some() {
                st.post_decision_pairs += 1;
                continue;
            }
            st.sprt.record_pair(0, 2);
        }
        assert_eq!(st.decided, Some(SprtVerdict::H1Accepted));
        assert_eq!(st.sprt.pairs(), pairs_at_decision);
        assert_eq!(st.sprt.llr(), llr_at_decision);
        assert_eq!(st.post_decision_pairs, 64);
    }

    #[test]
    fn pair_scoring_is_swap_symmetric() {
        use crate::versus::{EndReason, VersusOutcome, VersusResult};
        let out = |result| VersusOutcome {
            seed: 1,
            result,
            plies: 10,
            attack_a: 0,
            attack_b: 0,
            a_topped: false,
            b_topped: false,
            end_reason: EndReason::Topout,
        };
        // A wins the forward game as seat A and the reverse game as seat B.
        let pair = PairOutcome {
            seed: 1,
            forward: out(VersusResult::AWins),
            reverse: out(VersusResult::BWins),
        };
        assert_eq!(pair.a_wins_losses(), (2, 0));
        let split = PairOutcome {
            seed: 1,
            forward: out(VersusResult::AWins),
            reverse: out(VersusResult::AWins),
        };
        assert_eq!(split.a_wins_losses(), (1, 1));
    }
}
