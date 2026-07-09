//! Self-play datagen driver (leapfrog map ticket T14).
//!
//! Plays mirror self-play under the sudden-death venue with a policy-guided beam,
//! capturing every decision as a `DecisionRecord` (served children obs + the beam's
//! per-root backed-up score = the completed-Q source) and writing game-aligned
//! shards. The move actually applied is the beam's argmax (v0: no ε-sampling), so
//! `played == argmax` and the capture is unambiguous — no offline reconstruction,
//! no dependence on controller tie-break RNG.
//!
//! Same code, two uses: a **CC2** evaluator makes the round-0 BC corpus (CC2
//! rollouts → shards); a **net** evaluator makes round-1+ self-play. The driver
//! drives the `BeamPlanner` directly (to read `root_scores`) and applies the
//! chosen placement via `placement_to_inputs` + a replay controller through the
//! shared `versus_step_piece`, so the versus rules stay the engine's.

use tetr_core::ai::eval::Evaluator;
use tetr_core::ai::search::{hold_placements, think_to_completion};
use tetr_core::ai::state::SearchState;
use tetr_core::ai::{BeamPlanner, SearchBudget, placement_to_inputs};
use tetr_core::engine::{Engine, EngineSnapshot, InputFrame};
use tetr_core::player::PlayerController;
use tetr_nn::obs::{OppCtx, encode};
use tetr_nn::shards::{DecisionMeta, DecisionRecord, ShardWriter};

use tetr_core::engine::EngineEvent;

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

/// Beam config for the datagen bot.
#[derive(Clone, Copy)]
pub struct BeamConfig {
    pub width: usize,
    pub depth: u8,
    pub transpose: bool,
    /// Guided-vehicle restriction: net policy top-m placements per node
    /// (0 = unrestricted). Requires a net dir (the filter's ranker).
    pub top_m: usize,
}

fn planner(cfg: BeamConfig, net_dir: Option<&std::path::Path>) -> BeamPlanner {
    let base = if cfg.transpose {
        BeamPlanner::transposing(cfg.width)
    } else {
        BeamPlanner::new(cfg.width)
    };
    match (cfg.top_m, net_dir) {
        (m, Some(dir)) if m > 0 => base.with_root_filter(crate::arm::guided_filter(dir, m)),
        _ => base,
    }
}

/// One seat's decision: run the beam on `state`, capture the record (aligned to
/// `hold_placements`), and apply the argmax placement to `engine` via a replay
/// controller. Returns `(record, attack, topped)`; `record` is `None` only for a
/// topped-out state (no legal placement).
fn play_decision(
    engine: &mut Engine,
    beam: &mut BeamPlanner,
    eval: &dyn Evaluator,
    depth: u8,
    state: &SearchState,
    meta: DecisionMeta,
    opp: &OppCtx,
) -> (Option<DecisionRecord>, u32, bool) {
    if hold_placements(state).is_empty() {
        return (None, 0, true);
    }
    if std::env::var("TETR_DATAGEN_TRACE").is_ok() {
        let mut miny = [99i32; 10];
        let arr = state.board.to_array2d();
        for x in 0..10 {
            for y in 0..arr.height() {
                if arr.get_cell_kind(x as isize, y as isize).is_some() {
                    miny[x] = y as i32;
                    break;
                }
            }
        }
        eprintln!(
            "DTRACE seat={} n={} piece={:?} miny={:?}",
            meta.seat,
            meta.ply,
            state.active.piece_type(),
            miny
        );
    }
    think_to_completion(beam, state, eval, SearchBudget::beam(depth));

    // The decision's placements ARE the beam's roots — under a placement
    // filter that is the filtered subset, in the beam's own order. Deriving
    // them from `root_scores()` keeps placements and scores aligned BY
    // CONSTRUCTION (round-1 postmortem: indexing the full `hold_placements`
    // list with filtered-root scores mislabeled 36% of the corpus and
    // misplayed every game — policy collapsed 0-64).
    let (placements, scores): (Vec<_>, Vec<i32>) =
        beam.root_scores().map(|(p, s)| (p.clone(), s)).unzip();
    if placements.is_empty() {
        return (None, 0, true);
    }
    // FIRST maximum wins on ties — the planner's own back-up rule (`>`), and
    // load-bearing: CC2 integer evals tie on ~55% of decisions, and
    // `max_by_key` (last max) made the CC2 seat play systematically different,
    // pathological moves vs real CC2 (round-6 postmortem: lost ~1200-0 to the
    // net while the trusted duel says ~even).
    let mut argmax = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[argmax] {
            argmax = i;
        }
    }

    // Served children: the resulting state after each placement, encoded as the
    // net sees it (opponent-blind, matching the net arms).
    let child_obs: Vec<_> = placements
        .iter()
        .map(|p| {
            let mut child = state.clone();
            child.commit_placement(p);
            encode(&child, opp)
        })
        .collect();
    let children: Vec<_> = child_obs
        .iter()
        .zip(&scores)
        .zip(&placements)
        .map(|((o, &s), p)| (o, s, tetr_nn::obs::placement_slot(p)))
        .collect();
    let parent = encode(state, opp);
    let record = DecisionRecord::from_served(
        DecisionMeta {
            played: argmax as u16,
            argmax: argmax as u16,
            ..meta
        },
        &parent,
        &children,
    );

    // Apply the argmax placement (== best plan) to the engine.
    // Mirror the controller: render on the engine Board (BitBoard→array2d) from
    // the search state's active pose — the pose the movegen path was recorded from.
    let best = &placements[argmax];
    let frames = placement_to_inputs(&state.board.to_array2d(), &state.active, best);
    let mut replay = ReplayController {
        frames: frames.into_iter(),
    };
    let (attack, topped) = versus_step_piece(engine, &mut replay);
    (Some(record), attack, topped)
}

/// Play one mirror self-play game (both seats = the same beam config + eval),
/// pushing every decision to `writer` and sealing the game with its outcome.
/// Returns the outcome.
pub fn datagen_game(
    writer: &mut ShardWriter,
    eval: &dyn Evaluator,
    cfg: BeamConfig,
    net_dir: Option<&std::path::Path>,
    venue: &VersusFormat,
    seed: u64,
    game_id: u32,
) -> std::io::Result<VersusOutcomeLite> {
    datagen_game_vs(
        writer,
        [eval, eval],
        cfg,
        net_dir,
        false,
        venue,
        seed,
        game_id,
    )
}

/// Two-arm variant: when `opp_cc2` is set, one seat plays a plain TP beam with
/// `evals[1]` (a CC2 evaluator) at the same width/depth — grounded-opponent
/// games (long, competitive, value-rich) vs the degenerate short mirrors.
/// Which seat gets the net alternates by game parity so the data covers both.
/// BOTH seats' decisions are recorded (the CC2 seat's rows are r0-grade
/// grounded data; policy qnorm is per-decision scale-free so mixed score
/// units are safe for the current recipe — only the quarantined boot-value
/// would care).
pub fn datagen_game_vs(
    writer: &mut ShardWriter,
    evals: [&dyn Evaluator; 2],
    cfg: BeamConfig,
    net_dir: Option<&std::path::Path>,
    opp_cc2: bool,
    venue: &VersusFormat,
    seed: u64,
    game_id: u32,
) -> std::io::Result<VersusOutcomeLite> {
    let opp = OppCtx::default();
    let mut engines = [
        Engine::new(marathon_config(), seed),
        Engine::new(marathon_config(), seed),
    ];
    // Seat of the net arm alternates by game parity in two-arm mode.
    let net_seat: usize = if opp_cc2 { (game_id % 2) as usize } else { 0 };
    let seat_eval = |who: usize| -> &dyn Evaluator {
        if !opp_cc2 || who == net_seat {
            evals[0]
        } else {
            evals[1]
        }
    };
    let mut beams = if opp_cc2 {
        let net_beam = planner(cfg, net_dir);
        let cc2_beam = BeamPlanner::transposing(cfg.width);
        if net_seat == 0 {
            [net_beam, cc2_beam]
        } else {
            [cc2_beam, net_beam]
        }
    } else {
        [planner(cfg, net_dir), planner(cfg, net_dir)]
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
        // seat-A win skew over 1900 short mirror games — pure z-label noise.
        let order = if (ply + game_id) % 2 == 0 {
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
            // Split borrows: eval is shared &dyn, engines/beams indexed distinctly.
            let (engine_who, beam_who) = (&mut engines[who], &mut beams[who]);
            let (record, atk, topout) = play_decision(
                engine_who,
                beam_who,
                seat_eval(who),
                cfg.depth,
                &state,
                meta,
                &opp,
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
    /// verify the pipeline (round-trip, per-decision children align, z ∈ {−1,0,1}).
    /// Run: cargo test --release -p tetr-research --test-threads=1 datagen_writes_shards -- --nocapture
    #[test]
    fn datagen_writes_shards() {
        let dir = std::env::temp_dir().join(format!("tetr-datagen-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
        let cfg = BeamConfig {
            width: 6,
            depth: 4,
            transpose: true,
            top_m: 0,
        };
        // Pressured venue so games end fast.
        let venue = VersusFormat {
            max_plies: 60,
            rain_period: 4,
            sudden_death: true,
        };

        let mut total_games = 0;
        let mut total_decisions = 0;
        {
            let mut writer = ShardWriter::create(&dir, 64).expect("writer");
            for seed in 1..=4u64 {
                let out = datagen_game(&mut writer, &eval, cfg, None, &venue, seed, seed as u32)
                    .expect("game writes");
                total_games += 1;
                assert!(out.plies_total > 0, "game {seed} made no moves");
            }
            writer.flush().expect("final flush");
        }

        // Read back every shard and validate structure.
        let mut z_seen = [false; 3]; // -1, 0, +1
        for shard_path in std::fs::read_dir(&dir).unwrap().filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()?.to_str()? == "safetensors").then_some(p)
        }) {
            let shard = Shard::read(&shard_path).expect("shard loads + checksum ok");
            for (d_idx, meta) in shard.decisions.iter().enumerate() {
                total_decisions += 1;
                let n_children = shard.children_of(d_idx).len();
                assert!(n_children > 0, "decision has no children");
                assert!((meta.played as usize) < n_children, "played in range");
                assert!((meta.argmax as usize) < n_children, "argmax in range");
                assert!((-1..=1).contains(&meta.z), "z ∈ [-1,1]");
                z_seen[(meta.z + 1) as usize] = true;
            }
        }
        eprintln!(
            "\ndatagen: {total_games} games, {total_decisions} decisions; z_seen(-1,0,+1)={z_seen:?}"
        );
        assert!(total_decisions > 0, "no decisions written");
        // A decisive mirror set should show wins and losses (not all draws).
        assert!(
            z_seen[0] && z_seen[2],
            "expected both win and loss z labels"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Round-1 postmortem regression: with the guided filter active, the
    /// record's children must be the beam's FILTERED roots (<= top_m), and the
    /// game must play through — previously the full placement list was stored
    /// with misaligned filtered scores (corrupt labels + misplayed games).
    #[test]
    fn filtered_datagen_stores_the_beams_roots() {
        let dir = std::env::temp_dir().join(format!("tetr-datagen-filt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let net_dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tetr-nn/tests/fixtures/round0"
        ));
        let eval = tetr_nn::serve::NetEvaluator::load(net_dir).expect("fixture net");
        let cfg = BeamConfig {
            width: 6,
            depth: 3,
            transpose: true,
            top_m: 6,
        };
        let venue = VersusFormat {
            max_plies: 30,
            rain_period: 4,
            sudden_death: true,
        };
        let mut writer = ShardWriter::create(&dir, 8).expect("writer");
        let out = datagen_game(&mut writer, &eval, cfg, Some(net_dir), &venue, 7, 7).unwrap();
        writer.flush().unwrap();
        assert!(out.plies_total > 0);
        let mut checked = 0;
        for p in std::fs::read_dir(&dir).unwrap() {
            let p = p.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }
            let shard = Shard::read(&p).unwrap();
            for d in 0..shard.decisions.len() {
                let n = shard.children_of(d).len();
                assert!(n <= 6, "decision stored {n} children; filter is top-6");
                let scores = &shard.child_scores[shard.children_of(d)];
                let played = shard.decisions[d].played as usize;
                assert_eq!(
                    scores[played],
                    *scores.iter().max().unwrap(),
                    "played must be the argmax of the SERVED scores"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no decisions written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Driver throughput at a realistic config (CC2, w8d5, full venue) — the
    /// "measure the real bottleneck" step for T13. Reports games/hr so the driver
    /// overhead can be compared to the raw duel rate.
    /// Run: cargo test --release -p tetr-research datagen_throughput -- --ignored --nocapture
    #[test]
    #[ignore]
    fn datagen_throughput() {
        use std::time::Instant;
        let dir = std::env::temp_dir().join(format!("tetr-datagen-tput-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
        let cfg = BeamConfig {
            width: 8,
            depth: 5,
            transpose: true,
            top_m: 0,
        };
        let venue = VersusFormat {
            max_plies: 240,
            rain_period: 8,
            sudden_death: true,
        };
        let mut writer = ShardWriter::create(&dir, 1024).expect("writer");
        let n = 8u64;
        let t0 = Instant::now();
        for seed in 1..=n {
            let _ = datagen_game(&mut writer, &eval, cfg, None, &venue, seed, seed as u32).unwrap();
        }
        writer.flush().unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let mut decisions = 0usize;
        for p in std::fs::read_dir(&dir).unwrap() {
            let p = p.unwrap().path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                decisions += Shard::read(&p).unwrap().decisions.len();
            }
        }
        eprintln!(
            "\ndatagen CC2 w8d5 (single-thread): {n} games in {secs:.1}s = {:.0} games/hr, {decisions} decisions ({:.0} dec/s)",
            n as f64 * 3600.0 / secs,
            decisions as f64 / secs,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod divergence_probe {
    use super::*;
    use crate::arm::Arm;
    use std::str::FromStr;

    /// Wraps a controller and logs column min-y at each new-piece moment —
    /// the harness half of the divergence diff (the driver half is
    /// TETR_DATAGEN_TRACE in play_decision).
    struct Tracing {
        inner: Box<dyn PlayerController>,
        tag: &'static str,
        last: Option<(tetr_core::engine::PieceType, usize)>,
        n: u32,
    }

    impl PlayerController for Tracing {
        fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
            if let Some(active) = &snapshot.active {
                let sig = (active.piece_type, snapshot.board_cells.len());
                if self.last != Some(sig) {
                    self.last = Some(sig);
                    let mut miny = [99i32; 10];
                    for c in &snapshot.board_cells {
                        let x = c.x as usize;
                        if x < 10 {
                            miny[x] = miny[x].min(c.y as i32);
                        }
                    }
                    eprintln!(
                        "HTRACE {} n={} piece={:?} miny={:?}",
                        self.tag, self.n, active.piece_type, miny
                    );
                    self.n += 1;
                }
            }
            self.inner.poll(snapshot)
        }
    }

    /// Datagen defect #3 probe: same arm, same seed — harness game vs driver
    /// game. Prints ply counts; run with --nocapture. The driver should match
    /// the harness (both deterministic, reaction 0, blocking venue).
    /// cargo test --release -p tetr-research divergence -- --ignored --nocapture
    #[test]
    #[ignore]
    fn driver_vs_harness_same_seed() {
        // The v3 net via env (the fixture net's games are much longer — nets
        // differ in style; the probe must compare like with like).
        let dir_owned = std::env::var("PROBE_NET").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tetr-nn/tests/fixtures/round0"
            )
            .into()
        });
        let dir = dir_owned.as_str();
        let venue = VersusFormat {
            max_plies: 240,
            rain_period: 8,
            sudden_death: true,
        };
        for seed in [989000000u64] {
            // Harness path: the trusted versus loop with controller factories.
            let arm = Arm::from_str(&format!("guided:{dir}@m12w8d5")).unwrap();
            let fa0 = arm.factory();
            let fb0 = arm.factory();
            let fa = move |s: u64| -> Box<dyn PlayerController> {
                Box::new(Tracing {
                    inner: fa0(s),
                    tag: "A",
                    last: None,
                    n: 0,
                })
            };
            let fb = move |s: u64| -> Box<dyn PlayerController> {
                Box::new(Tracing {
                    inner: fb0(s),
                    tag: "B",
                    last: None,
                    n: 0,
                })
            };
            let out = crate::versus::play_versus_format(&fa, &fb, seed, venue);
            eprintln!(
                "harness seed {seed}: plies={} result={:?} end={:?}",
                out.plies, out.result, out.end_reason
            );

            // Driver path: same arm config through datagen_game.
            let tmp = std::env::temp_dir().join(format!("tetr-divergence-{seed}"));
            let _ = std::fs::remove_dir_all(&tmp);
            let eval = tetr_nn::serve::NetEvaluator::load(dir).expect("net");
            let cfg = BeamConfig {
                width: 8,
                depth: 5,
                transpose: true,
                top_m: 12,
            };
            let mut writer = ShardWriter::create(&tmp, 100_000).unwrap();
            let out2 = datagen_game(
                &mut writer,
                &eval,
                cfg,
                Some(std::path::Path::new(dir)),
                &venue,
                seed,
                seed as u32,
            )
            .unwrap();
            eprintln!(
                "driver  seed {seed}: plies={} result={:?} end={:?}",
                out2.plies_total, out2.result, out2.end_reason
            );
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }
}
