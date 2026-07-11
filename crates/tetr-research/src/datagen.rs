//! Self-play datagen: play mirror games, record what was played, label with
//! the outcome.
//!
//! Both seats run the same beam + evaluator (a CC2 evaluator makes the round-0
//! bootstrap corpus; a net evaluator makes every later round's self-play).
//! Each decision stores ONE row — the served observation of the state the
//! mover chose — and the game's outcome backfills `z` at the end. The move
//! applied is the beam's argmax, so the capture is unambiguous: no sampling,
//! no offline reconstruction.
//!
//! The driver drives the `BeamPlanner` directly and applies the chosen
//! placement via `placement_to_inputs` + a replay controller through the
//! shared `versus_step_piece`, so the versus rules stay the engine's — a
//! seed-matched driver game reproduces a harness (duel) game ply for ply.

use tetr_core::ai::eval::Evaluator;
use tetr_core::ai::search::{hold_placements, think_to_completion};
use tetr_core::ai::state::SearchState;
use tetr_core::ai::{BeamPlanner, SearchBudget, placement_to_inputs};
use tetr_core::engine::{Engine, EngineEvent, EngineSnapshot, InputFrame};
use tetr_core::player::PlayerController;
use tetr_nn::obs::encode;
use tetr_nn::shards::{DecisionMeta, DecisionRecord, ShardWriter};

use crate::marathon::marathon_config;
use crate::versus::{EndReason, VersusFormat, VersusResult, decide_versus, versus_step_piece};

/// The engine's nominal idle timestep (mirrors the controller's `neutral()`).
const NOMINAL_DT: f32 = 1.0 / 60.0;

/// Step `engine` with idle frames (dt>0, so ARE/gravity advance) until a piece is
/// active (spawn complete), returning its `SearchState`. `None` = the game topped
/// out at spawn. A fresh engine has no active piece until stepped, and maneuver
/// frames carry `dt==0`, so this is how a piece gets on the board before planning.
fn advance_to_active(engine: &mut Engine) -> Option<SearchState> {
    for _ in 0..600 {
        if let Some(state) = SearchState::from_snapshot(&engine.snapshot()) {
            return Some(state);
        }
        let idle = InputFrame {
            dt_seconds: NOMINAL_DT,
            ..InputFrame::default()
        };
        if engine
            .step(idle)
            .iter()
            .any(|e| matches!(e, EngineEvent::GameOver { .. }))
        {
            return None;
        }
    }
    None
}

/// Feeds a pre-computed input sequence one frame per poll; neutral once drained
/// (the piece has already hard-dropped, so trailing polls are no-ops).
struct ReplayController {
    frames: std::vec::IntoIter<InputFrame>,
}

impl PlayerController for ReplayController {
    fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
        self.frames.next().unwrap_or_default()
    }
}

/// Beam config for the datagen bot — the SAME plain beam the `beam:` arms
/// race, so the data plant and the gate measure one vehicle (a transposing
/// datagen beam vs a plain gate beam once diverged play by 2 plies in 214;
/// the driver≡harness test below pins them together).
#[derive(Clone, Copy)]
pub struct BeamConfig {
    pub width: usize,
    pub depth: u8,
}

fn planner(cfg: BeamConfig) -> BeamPlanner {
    BeamPlanner::new(cfg.width)
}

/// One seat's decision: run the beam on `state`, record the played state's
/// served observation, and apply the argmax placement to `engine` via a replay
/// controller. Returns `(record, attack, topped)`; `record` is `None` only for
/// a topped-out state (no legal placement).
fn play_decision(
    engine: &mut Engine,
    beam: &mut BeamPlanner,
    eval: &dyn Evaluator,
    depth: u8,
    state: &SearchState,
    meta: DecisionMeta,
) -> (Option<DecisionRecord>, u32, bool) {
    if hold_placements(state).is_empty() {
        return (None, 0, true);
    }
    think_to_completion(beam, state, eval, SearchBudget::beam(depth));

    // The decision's placements ARE the beam's roots, in the beam's own order
    // — deriving them from `root_scores()` keeps placements and scores aligned
    // BY CONSTRUCTION (a past driver indexed one list with the other's
    // positions and mislabeled a third of a corpus).
    let (placements, scores): (Vec<_>, Vec<i32>) =
        beam.root_scores().map(|(p, s)| (p.clone(), s)).unzip();
    if placements.is_empty() {
        return (None, 0, true);
    }
    // FIRST maximum wins on ties — the planner's own back-up rule (`>`), and
    // load-bearing: CC2 integer evals tie on ~55% of decisions, and a last-max
    // argmax made the driver play systematically differently from the duel
    // harness on identical states.
    let mut argmax = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[argmax] {
            argmax = i;
        }
    }

    // Record the played (post-placement) state exactly as the net would see
    // it, plus one deterministically-chosen NON-best sibling (the ranking
    // pair: "the search preferred played over alt"). Deterministic pick — no
    // RNG in the driver, reproducible from (seed, ply).
    let best = &placements[argmax];
    let mut played = state.clone();
    played.commit_placement(best);
    let alt_obs = (placements.len() > 1).then(|| {
        let k = (meta.game_id as usize)
            .wrapping_mul(31)
            .wrapping_add(meta.ply as usize * 7 + meta.seat as usize)
            % (placements.len() - 1);
        let alt_idx = if k >= argmax { k + 1 } else { k };
        let mut alt = state.clone();
        alt.commit_placement(&placements[alt_idx]);
        encode(&alt)
    });
    let record = DecisionRecord::from_served(meta, &encode(&played), alt_obs.as_ref());

    // Apply the placement to the engine. Mirror the controller: render on the
    // engine Board (BitBoard→array2d) from the search state's active pose —
    // the pose the movegen path was recorded from.
    let frames = placement_to_inputs(&state.board.to_array2d(), &state.active, best);
    let mut replay = ReplayController {
        frames: frames.into_iter(),
    };
    let (attack, topped) = versus_step_piece(engine, &mut replay);
    (Some(record), attack, topped)
}

/// Play one self-play game, pushing every decision to `writer` and sealing
/// the game with its outcome. Both seats share the evaluator; `opp_width`
/// (when set) gives one seat a NARROWER beam — the unbalanced-pair teacher
/// mode, which makes mid-game boards outcome-predictive (balanced mirror
/// games are decided late by piece luck, so their z labels are coin flips
/// for the first ~60% of a game — measured 2026-07-10). Which seat gets the
/// wide beam alternates by game parity so z stays seat-balanced.
pub fn datagen_game(
    writer: &mut ShardWriter,
    eval: &dyn Evaluator,
    cfg: BeamConfig,
    opp_width: Option<usize>,
    venue: &VersusFormat,
    seed: u64,
    game_id: u32,
) -> std::io::Result<VersusOutcomeLite> {
    let mut engines = [
        Engine::new(marathon_config(), seed),
        Engine::new(marathon_config(), seed),
    ];
    let narrow = BeamConfig {
        width: opp_width.unwrap_or(cfg.width),
        depth: cfg.depth,
    };
    let mut beams = if game_id.is_multiple_of(2) {
        [planner(cfg), planner(narrow)]
    } else {
        [planner(narrow), planner(cfg)]
    };
    let mut attack = [0u32; 2];
    let mut topped = [false; 2];
    let mut ply_of = [0u16; 2];
    let mut end_ply = venue.hard_cap();

    'game: for ply in 0..venue.hard_cap() {
        let period = venue.rain_period_at(ply);
        if period > 0 && ply % period == period - 1 {
            engines[0].queue_garbage(1);
            engines[1].queue_garbage(1);
        }
        // Alternate first mover per ply AND stagger which seat opens the game
        // by game parity: with one game per seed (no arm-swapped CRN pair like
        // the duel instrument), a fixed ply-0 opener left a measured ~5σ
        // seat-A win skew over 1900 mirror games — pure z-label noise.
        let order = if (ply + game_id).is_multiple_of(2) {
            [0usize, 1]
        } else {
            [1, 0]
        };
        for &who in &order {
            let Some(state) = advance_to_active(&mut engines[who]) else {
                topped[who] = true;
                end_ply = ply;
                break 'game;
            };
            let meta = DecisionMeta {
                game_id,
                seat: who as u8,
                ply: ply_of[who],
                ..Default::default()
            };
            ply_of[who] += 1;
            let (record, atk, topout) = play_decision(
                &mut engines[who],
                &mut beams[who],
                eval,
                cfg.depth,
                &state,
                meta,
            );
            if let Some(r) = record {
                writer.push(r);
            }
            if atk > 0 {
                engines[who ^ 1].queue_garbage(atk);
                attack[who] += atk;
            }
            if topout {
                topped[who] = true;
                end_ply = ply;
                break 'game;
            }
        }
    }

    let result = decide_versus(topped[0], topped[1], attack[0], attack[1]);
    let end_reason = if topped[0] || topped[1] {
        if venue.sudden_death && end_ply >= venue.max_plies {
            EndReason::Escalation
        } else {
            EndReason::Topout
        }
    } else {
        EndReason::TrueCap
    };
    let plies_total = ply_of[0] + ply_of[1];
    let z_for_seat = |seat: u8| -> i8 {
        match (seat, result) {
            (0, VersusResult::AWins) | (1, VersusResult::BWins) => 1,
            (0, VersusResult::BWins) | (1, VersusResult::AWins) => -1,
            _ => 0,
        }
    };
    writer.finish_game(z_for_seat, end_reason as u8, plies_total)?;

    Ok(VersusOutcomeLite {
        result,
        end_reason,
        plies_total,
        attack,
    })
}

/// The datagen game's outcome (the shard already has the per-decision z).
#[derive(Debug, Clone, Copy)]
pub struct VersusOutcomeLite {
    pub result: VersusResult,
    pub end_reason: EndReason,
    pub plies_total: u16,
    pub attack: [u32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetr_core::ai::Cc2Evaluator;
    use tetr_core::ai::eval::Cc2Weights;
    use tetr_nn::shards::Shard;

    /// End-to-end: a few CC2-beam mirror games write shards; read them back and
    /// verify the pipeline (round-trip, z ∈ {−1,0,1}, both outcomes seen).
    #[test]
    fn datagen_writes_shards() {
        let dir = std::env::temp_dir().join(format!("tetr-datagen-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
        let cfg = BeamConfig { width: 6, depth: 4 };
        // Pressured venue so games end fast.
        let venue = VersusFormat {
            max_plies: 60,
            rain_period: 4,
            sudden_death: true,
        };

        let mut total_decisions = 0;
        {
            let mut writer = ShardWriter::create(&dir, 64).expect("writer");
            for seed in 1..=4u64 {
                let out = datagen_game(&mut writer, &eval, cfg, None, &venue, seed, seed as u32)
                    .expect("game writes");
                assert!(out.plies_total > 0, "game {seed} made no moves");
            }
            writer.flush().expect("final flush");
        }

        let mut z_seen = [false; 3]; // -1, 0, +1
        for shard_path in std::fs::read_dir(&dir).unwrap().filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()?.to_str()? == "safetensors").then_some(p)
        }) {
            let shard = Shard::read(&shard_path).expect("shard loads + checksum ok");
            for meta in &shard.decisions {
                total_decisions += 1;
                assert!((-1..=1).contains(&meta.z), "z ∈ [-1,1]");
                z_seen[(meta.z + 1) as usize] = true;
            }
            assert_eq!(shard.decisions.len(), shard.own.len());
            assert_eq!(shard.decisions.len(), shard.feats.len());
        }
        assert!(total_decisions > 0, "no decisions written");
        // A decisive mirror set should show wins and losses (not all draws).
        assert!(
            z_seen[0] && z_seen[2],
            "expected both win and loss z labels"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Driver ≡ harness: the same arm at the same seed must play the same game
    /// through the datagen driver as through the duel harness's versus loop.
    /// A silent divergence between these two paths (different tie-break,
    /// different vehicle, different frame stepping) has corrupted corpora
    /// before — this pins ply count and outcome on a real game.
    #[test]
    fn driver_matches_the_harness_at_the_same_seed() {
        use crate::arm::Arm;
        use std::str::FromStr;

        let venue = VersusFormat {
            max_plies: 60,
            rain_period: 4,
            sudden_death: true,
        };
        let seed = 42u64;

        // Harness path: the trusted versus loop with controller factories.
        let arm = Arm::from_str("beam:cc2@w6d4").unwrap();
        let out_h = crate::versus::play_versus_format(&arm.factory(), &arm.factory(), seed, venue);

        // Driver path: same config through datagen_game.
        let tmp = std::env::temp_dir().join(format!("tetr-divergence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
        let cfg = BeamConfig { width: 6, depth: 4 };
        let mut writer = ShardWriter::create(&tmp, 100_000).unwrap();
        let out_d = datagen_game(&mut writer, &eval, cfg, None, &venue, seed, seed as u32).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(out_h.plies, out_d.plies_total as u32, "ply count diverged");
        assert_eq!(format!("{:?}", out_h.result), format!("{:?}", out_d.result));
    }
}
