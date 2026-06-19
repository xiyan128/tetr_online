//! E0 (roadmap §2.1): does search depth past the ~6-ply preview horizon change the DECISION?
//!
//! The scaling sweep showed depth's Elo advantage over width collapses from 6.9x (concrete,
//! d<=6) to 1.9x (speculative, d>=7). If the steep returns are an artifact of the concrete
//! preview, the chosen ply-1 move should STABILIZE by ~d6-d7 and deeper plies should not move
//! it. This probe runs the champion-family TP-beam at depth 1..MAX over a bank of realistic
//! mid-game states and records, per state, the depth at which `best()`'s argmax stops changing.
//!
//! Decisive read:
//!  - if most states freeze by d6-d7 and d>=9 almost never changes the move -> deeper search is
//!    decoration; the expensive E1 tournament is not worth running.
//!  - if the move keeps flipping past d7 (and score keeps improving) -> depth is a real lever;
//!    escalate to E1 (narrow-deep vs champion under rain GSPRT).
//!
//! Run: cargo run --release -p tetr-research --example depth_probe

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights};
use tetr_core::ai::{BeamPlanner, SearchBudget, SearchState, think_to_completion};
use tetr_core::engine::{Engine, EngineEvent};
use tetr_core::player::drive_engine;
use tetr_research::bots::BotSpec;
use tetr_research::marathon::marathon_config;

const WIDTHS: &[usize] = &[8, 16, 32];
const MAX_DEPTH: u8 = 15;
const N_STATES: usize = 60;
const STATE_SEED: u64 = 0x0E10_0BEE;

/// The ply-1 decision identity: the chosen placement's pose + whether it used a hold swap.
/// Two decisions are "the same move" iff these match.
type Decision = (isize, isize, u8, bool);

fn decision(
    s: &SearchState,
    width: usize,
    depth: u8,
    eval: &Cc2Evaluator,
) -> Option<(Decision, i32)> {
    let mut mind = BeamPlanner::transposing(width);
    let plan = think_to_completion(&mut mind, s, eval, SearchBudget::beam(depth))?;
    let p = &plan.placement;
    let (ox, oy) = p.piece.origin();
    Some(((ox, oy, p.piece.rotation() as u8, p.used_hold), plan.score))
}

fn representative_states(n: usize) -> Vec<SearchState> {
    let mut engine = Engine::new(marathon_config(), STATE_SEED);
    let mut bot = BotSpec::tp_beam(16, 4)
        .cc2(Cc2Weights::attack_tuned())
        .factory()(STATE_SEED);
    let mut states = Vec::new();
    'outer: while states.len() < n {
        let snap = engine.snapshot();
        if snap.game_over.is_some() {
            break;
        }
        if let Some(s) = SearchState::from_snapshot(&snap) {
            // skip near-empty early boards (no real decision pressure)
            if states.len() < n {
                states.push(s);
            }
        }
        for _ in 0..4000 {
            let mut locked = false;
            for ev in drive_engine(&mut engine, &mut *bot) {
                match ev {
                    EngineEvent::Locked { .. } => locked = true,
                    EngineEvent::GameOver { .. } => break 'outer,
                    _ => {}
                }
            }
            if locked {
                break;
            }
        }
    }
    states
}

fn main() {
    let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
    let states = representative_states(N_STATES);
    let queue_len = states.iter().map(|s| s.queue.len()).max().unwrap_or(0);
    let concrete = queue_len + 1; // active piece + revealed previews = concrete plies
    eprintln!(
        "{} mid-game states; revealed queue = {} previews -> {} CONCRETE plies (depth > {} is speculative)",
        states.len(),
        queue_len,
        concrete,
        concrete
    );
    println!(
        "\nE0: depth at which the ply-1 decision stabilizes (TP-beam, attack-tuned). concrete horizon = d{concrete}.\n"
    );
    println!(
        "{:>4}  {:>8} {:>8} {:>8}  {:>10} {:>10} {:>10}  {:>14}",
        "w", "med_stab", "mean", "p90_stab", "d6==final", "d9==final", "d>9 moves", "score d9->15"
    );

    for &width in WIDTHS {
        let mut stab = Vec::new(); // per-state stabilization depth
        let mut d6_eq = 0; // final move already chosen by d6 (concrete horizon)
        let mut d9_eq = 0; // final move already chosen by d9 (old grid cap)
        let mut past9_moves = 0; // the move CHANGES somewhere in d10..15
        let mut score_gain_9_15 = 0.0f64; // mean score improvement d9 -> d15 (eval units)
        let mut n_gain = 0;

        for s in &states {
            let mut decisions = Vec::with_capacity(MAX_DEPTH as usize);
            let mut topped = false;
            for d in 1..=MAX_DEPTH {
                match decision(s, width, d, &eval) {
                    Some(x) => decisions.push(x),
                    None => {
                        topped = true;
                        break;
                    }
                }
            }
            if topped || decisions.len() < MAX_DEPTH as usize {
                continue; // no legal/decisive move at some depth — skip this state
            }
            let final_dec = decisions[MAX_DEPTH as usize - 1].0;
            // stabilization depth = 1 + last depth whose move differs from the final move
            let mut s_depth = 1u8;
            for (i, (dec, _)) in decisions.iter().enumerate() {
                if *dec != final_dec {
                    s_depth = (i as u8) + 2; // first depth AFTER this change
                }
            }
            s_depth = s_depth.min(MAX_DEPTH);
            stab.push(s_depth);
            if decisions[5].0 == final_dec {
                d6_eq += 1; // depth 6 (index 5) already == final
            }
            if decisions[8].0 == final_dec {
                d9_eq += 1; // depth 9 (index 8) already == final
            }
            // does the move change anywhere strictly after d9?
            if decisions[9..].iter().any(|(d, _)| *d != decisions[8].0) {
                past9_moves += 1;
            }
            let g = (decisions[MAX_DEPTH as usize - 1].1 - decisions[8].1) as f64;
            score_gain_9_15 += g;
            n_gain += 1;
        }

        stab.sort_unstable();
        let n = stab.len();
        let med = stab[n / 2];
        let mean = stab.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let p90 = stab[(n * 9 / 10).min(n - 1)];
        println!(
            "{:>4}  {:>8} {:>8.1} {:>8}  {:>9.0}% {:>9.0}% {:>9.0}%  {:>14.1}",
            width,
            med,
            mean,
            p90,
            100.0 * d6_eq as f64 / n as f64,
            100.0 * d9_eq as f64 / n as f64,
            100.0 * past9_moves as f64 / n as f64,
            score_gain_9_15 / n_gain.max(1) as f64,
        );
    }

    println!(
        "\nread: 'd6==final' = % of states whose deepest (d15) decision was already chosen at d6.\n\
         'd>9 moves' = % where the move still CHANGES somewhere past d9. High d6==final + low\n\
         d>9 moves => depth past the preview is decoration (refute E1). The opposite => run E1.\n\
         'score d9->15' is the mean eval-unit gain from 6 more plies (a stable move can still\n\
         improve its predicted value)."
    );
}
