//! `cc2-baseline` — measure **Cold Clear 2**'s attack-per-piece (APP) on the same
//! seeded task our bot plays, so we have a real, reproducible baseline to beat.
//!
//! We are the referee: CC2 (a TBP subprocess, see [`tetr_research::cc2`]) suggests
//! moves on a board it tracks itself; we mirror the board in a tiny bitboard sim,
//! drive the *same seeded 7-bag* (`tetr_core::engine::PieceGenerator`) our bot uses,
//! apply CC2's chosen cells, count the clears, and score them with the identical
//! [`tetr_core::engine::attack_lines`] table — using CC2's reported spin. Hold is
//! inferred from the placed piece (TBP semantics), with asserts guarding board sync.
//!
//! Run: `CC2_BIN=/path/to/cold-clear-2 SEEDS=6 PIECES=100 THINK_MS=50 \
//!       cargo run --release -p tetr-research --bin cc2-baseline`

use std::collections::VecDeque;
use std::time::Duration;

use tetr_core::engine::{attack_lines, EngineScoreAction, PieceGenerator, PieceType, TSpinKind};
use tetr_research::cc2::{Cc2, TbpBoard, TbpMove};
use tetr_research::versus_legacy::{versus_hole, GarbageQueue, VersusEngine};
use tetr_research::{
    beam_linear_bot, cheese_holes, decide_versus, evaluate_downstack, seed_set, VersusResult,
};

const WIDTH: i32 = 10;
const FULL_ROW: u16 = (1 << WIDTH) - 1;
const VISIBLE_QUEUE: usize = 7;

use tetr_research::cli::env_usize;

// --- TBP <-> piece mapping ---------------------------------------------------

fn piece_letter(p: PieceType) -> &'static str {
    match p {
        PieceType::I => "I",
        PieceType::J => "J",
        PieceType::L => "L",
        PieceType::O => "O",
        PieceType::S => "S",
        PieceType::T => "T",
        PieceType::Z => "Z",
    }
}

fn piece_from_letter(s: &str) -> PieceType {
    match s {
        "I" => PieceType::I,
        "J" => PieceType::J,
        "L" => PieceType::L,
        "O" => PieceType::O,
        "S" => PieceType::S,
        "T" => PieceType::T,
        "Z" => PieceType::Z,
        other => panic!("unknown TBP piece {other:?}"),
    }
}

// --- CC2's coordinate convention (verbatim from cold-clear-2 src/data.rs) -----

/// North-relative cells of each piece (its rotation center at the origin).
fn base_cells(p: PieceType) -> [(i32, i32); 4] {
    match p {
        PieceType::I => [(-1, 0), (0, 0), (1, 0), (2, 0)],
        PieceType::O => [(0, 0), (1, 0), (0, 1), (1, 1)],
        PieceType::T => [(-1, 0), (0, 0), (1, 0), (0, 1)],
        PieceType::L => [(-1, 0), (0, 0), (1, 0), (1, 1)],
        PieceType::J => [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        PieceType::S => [(-1, 0), (0, 0), (0, 1), (1, 1)],
        PieceType::Z => [(-1, 1), (0, 1), (0, 0), (1, 0)],
    }
}

/// CC2's `Rotation::rotate` applied to a (x, y) offset.
fn rotate(orientation: &str, (x, y): (i32, i32)) -> (i32, i32) {
    match orientation {
        "north" => (x, y),
        "east" => (y, -x),
        "south" => (-x, -y),
        "west" => (-y, x),
        other => panic!("unknown TBP orientation {other:?}"),
    }
}

/// The four absolute board cells CC2's move occupies (x in 0..10, y bottom-origin).
fn cc2_cells(mv: &TbpMove) -> [(i32, i32); 4] {
    let piece = piece_from_letter(&mv.location.piece);
    base_cells(piece).map(|c| {
        let (rx, ry) = rotate(&mv.location.orientation, c);
        (mv.location.x + rx, mv.location.y + ry)
    })
}

// --- a minimal board to count clears (occupancy only) ------------------------

struct Sim {
    rows: Vec<u16>, // rows[y], bit x set = filled; y bottom-origin, 40 tall
}

impl Sim {
    fn new() -> Self {
        Self {
            rows: vec![0u16; 40],
        }
    }

    /// A board pre-filled with `rows` of seeded cheese (each row full except its
    /// hole) — identical to what `tetr_research::play_downstack` paints, so both
    /// bots face the same garbage for a given seed.
    fn with_cheese(seed: u64, rows: usize) -> Self {
        let mut sim = Sim::new();
        for (y, &hole) in cheese_holes(seed, rows).iter().enumerate() {
            for x in 0..10usize {
                if x != hole {
                    sim.rows[y] |= 1 << x;
                }
            }
        }
        sim
    }

    /// Raise the stack by `lines` garbage rows (each full except `hole`), mirroring
    /// `tetr_core::engine::Board::insert_garbage_lines` so CC2 receives the same
    /// garbage shape our engine does. Returns true on overflow (top-out).
    fn insert_garbage(&mut self, lines: usize, hole: usize) -> bool {
        if lines == 0 {
            return false;
        }
        let overflow = self.rows[40usize.saturating_sub(lines)..]
            .iter()
            .any(|&r| r != 0);
        for y in (0..40).rev() {
            self.rows[y] = if y >= lines { self.rows[y - lines] } else { 0 };
        }
        let mut row = 0u16;
        for x in 0..10usize {
            if x != hole {
                row |= 1 << x;
            }
        }
        for y in 0..lines.min(40) {
            self.rows[y] = row;
        }
        overflow
    }

    /// Place the four cells, then clear full rows; returns lines cleared.
    fn place_and_clear(&mut self, cells: &[(i32, i32); 4]) -> u32 {
        for &(x, y) in cells {
            assert!(
                (0..WIDTH).contains(&x) && (0..40).contains(&y),
                "cell out of bounds: ({x},{y})"
            );
            self.rows[y as usize] |= 1 << x;
        }
        let before = self.rows.len();
        self.rows.retain(|&r| r != FULL_ROW);
        let cleared = (before - self.rows.len()) as u32;
        while self.rows.len() < 40 {
            self.rows.push(0);
        }
        cleared
    }

    fn is_empty(&self) -> bool {
        self.rows.iter().all(|&r| r == 0)
    }
}

fn empty_tbp_board() -> TbpBoard {
    vec![vec![None; WIDTH as usize]; 40]
}

/// A TBP board pre-filled with `rows` of seeded cheese ("G" garbage cells), matching
/// [`Sim::with_cheese`] so CC2 and our sim start from the identical garbage.
fn cheese_tbp_board(seed: u64, rows: usize) -> TbpBoard {
    let mut board = empty_tbp_board();
    for (y, &hole) in cheese_holes(seed, rows).iter().enumerate() {
        for (x, cell) in board[y].iter_mut().enumerate() {
            if x != hole {
                *cell = Some("G".to_string());
            }
        }
    }
    board
}

/// CC2's mirrored board as a TBP board for re-sync. Marks every occupied cell as
/// garbage ("G"): CC2 plays on occupancy + shape, so the loss of piece-colour
/// fidelity is a minor eval nuance, documented in [`run_versus`].
fn sim_to_tbp_board(sim: &Sim) -> TbpBoard {
    let mut board = empty_tbp_board();
    for (y, &bits) in sim.rows.iter().enumerate().take(40) {
        for (x, cell) in board[y].iter_mut().enumerate() {
            if bits & (1 << x) != 0 {
                // A normal piece colour, NOT "G": we don't track which cells are
                // garbage vs CC2's own stack, and marking the whole board "G" makes
                // CC2 treat its entire stack as garbage (frantic, low-quality play).
                // Occupancy + shape is what matters for legal placement.
                *cell = Some("I".to_string());
            }
        }
    }
    board
}

/// `EngineScoreAction` for CC2's reported (spin, lines) — feeds the attack table.
fn action_for(spin: &str, lines: usize) -> EngineScoreAction {
    match spin {
        "full" => EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines,
        },
        "mini" => EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines,
        },
        _ => match lines {
            1 => EngineScoreAction::Single,
            2 => EngineScoreAction::Double,
            3 => EngineScoreAction::Triple,
            4 => EngineScoreAction::Tetris,
            _ => EngineScoreAction::NoClear,
        },
    }
}

/// Whether a clear extends/starts a Back-to-Back chain.
fn b2b_eligible(spin: &str, lines: usize) -> bool {
    lines > 0 && (matches!(spin, "full" | "mini") || lines == 4)
}

/// Resolve TBP's implicit hold from the piece CC2 actually `placed`, advancing `queue`
/// and `hold` to match. TBP sends no explicit hold event: the placed piece is the queue
/// front (pop it), the held piece (swap the front into hold), or — on a first, empty-hold
/// swap — the *second* queue piece (park the front in hold; the placed one is what
/// followed). The asserts catch any queue/hold desync against CC2's own model.
fn advance_queue_hold(
    queue: &mut VecDeque<PieceType>,
    hold: &mut Option<PieceType>,
    placed: PieceType,
    seed: u64,
) {
    let active = queue[0];
    if placed == active {
        queue.pop_front();
    } else if let Some(h) = *hold {
        assert_eq!(placed, h, "hold-swap piece mismatch (seed {seed})");
        *hold = Some(queue.pop_front().unwrap());
    } else {
        let to_hold = queue.pop_front().unwrap();
        let next = queue.pop_front().unwrap();
        assert_eq!(
            placed, next,
            "empty-hold placed piece mismatch (seed {seed})"
        );
        *hold = Some(to_hold);
    }
}

/// Score one CC2 clear into attack lines, advancing the `combo` / `b2b` chain. `lines`
/// is the rows the placement cleared (0 ⇒ no clear: the combo breaks and no attack is
/// sent). The B2B eligibility is computed **once** and reused for both the bonus gate
/// (against the *incoming* `b2b`) and the new chain state. Returns the attack sent.
fn score_cc2_clear(sim: &Sim, spin: &str, lines: usize, combo: &mut u32, b2b: &mut bool) -> u32 {
    if lines == 0 {
        *combo = 0;
        return 0;
    }
    let eligible = b2b_eligible(spin, lines);
    let bonus = *b2b && eligible;
    let pc = sim.is_empty();
    let atk = attack_lines(action_for(spin, lines), bonus, *combo, pc);
    *combo += 1;
    *b2b = eligible;
    atk
}

/// Play CC2 over one seeded game of `pieces` placements; return total attack.
fn run_one(bin: &str, seed: u64, pieces: usize, think: Duration) -> std::io::Result<u32> {
    let mut gen = PieceGenerator::with_seed(seed);
    let mut queue: VecDeque<PieceType> = (0..VISIBLE_QUEUE).map(|_| gen.next().unwrap()).collect();
    let mut hold: Option<PieceType> = None;

    let mut cc2 = Cc2::spawn(bin)?;
    let queue_letters: Vec<String> = queue.iter().map(|&p| piece_letter(p).to_string()).collect();
    let bag: Vec<String> = PieceType::all()
        .iter()
        .map(|&p| piece_letter(p).to_string())
        .collect();
    cc2.start(&empty_tbp_board(), &queue_letters, None, 0, false, &bag)?;

    let mut sim = Sim::new();
    let mut combo = 0u32;
    let mut b2b = false;
    let mut total_attack = 0u32;

    for _ in 0..pieces {
        let Some(mv) = cc2.suggest(think)? else {
            break; // no legal move: CC2 forfeits (topped out)
        };
        let placed = piece_from_letter(&mv.location.piece);

        // Resolve hold from the placed piece (TBP infers hold this way).
        advance_queue_hold(&mut queue, &mut hold, placed, seed);

        // Apply CC2's placement + score the clear with our attack table.
        let lines = sim.place_and_clear(&cc2_cells(&mv)) as usize;
        total_attack += score_cc2_clear(&sim, &mv.spin, lines, &mut combo, &mut b2b);

        // Reveal the next piece (one per move keeps our queue in sync with CC2's).
        let revealed = gen.next().unwrap();
        queue.push_back(revealed);
        cc2.play(&mv, piece_letter(revealed))?;
    }

    Ok(total_attack)
}

/// Drive CC2 to clear `garbage_rows` of seeded cheese; return (pieces used, cleared?).
/// The digging metric — not gameable by combo-farming.
fn run_downstack(
    bin: &str,
    seed: u64,
    garbage_rows: u32,
    max_pieces: u32,
    think: Duration,
) -> std::io::Result<(u32, bool)> {
    let rows = garbage_rows as usize;
    let mut gen = PieceGenerator::with_seed(seed);
    let mut queue: VecDeque<PieceType> = (0..VISIBLE_QUEUE).map(|_| gen.next().unwrap()).collect();
    let mut hold: Option<PieceType> = None;

    let mut cc2 = Cc2::spawn(bin)?;
    let queue_letters: Vec<String> = queue.iter().map(|&p| piece_letter(p).to_string()).collect();
    let bag: Vec<String> = PieceType::all()
        .iter()
        .map(|&p| piece_letter(p).to_string())
        .collect();
    cc2.start(
        &cheese_tbp_board(seed, rows),
        &queue_letters,
        None,
        0,
        false,
        &bag,
    )?;

    let mut sim = Sim::with_cheese(seed, rows);
    let mut pieces = 0u32;
    let mut cleared_total = 0u32;

    while pieces < max_pieces {
        let Some(mv) = cc2.suggest(think)? else {
            break; // forfeit (topped out)
        };
        let placed = piece_from_letter(&mv.location.piece);

        advance_queue_hold(&mut queue, &mut hold, placed, seed);

        cleared_total += sim.place_and_clear(&cc2_cells(&mv));
        pieces += 1;
        if cleared_total >= garbage_rows {
            return Ok((pieces, true));
        }

        let revealed = gen.next().unwrap();
        queue.push_back(revealed);
        cc2.play(&mv, piece_letter(revealed))?;
    }

    Ok((pieces, cleared_total >= garbage_rows))
}

/// One **versus** match: our beam (A) vs CC2 (B), mutual garbage with cancellation.
/// Both face the identical piece stream (same seed). A player loses by topping out;
/// at the ply cap the higher total attack wins. CC2 is re-synced over TBP
/// (`stop`+`start`) whenever garbage lands on it — the portable way to inject
/// garbage into a base-TBP bot. Caveat: the re-sent board marks every occupied cell
/// as "G"; CC2 plays on occupancy/shape so the effect is minor but not perfect.
/// Caveat 2: this referee keeps the harness garbage rules (wholesale dump every
/// ply, oldest-lowest stacking) rather than the engine's guideline rules
/// (deferred, capped rising), so its win rates are NOT like-for-like with
/// `play_versus` — compare runs of this referee only with each other.
/// Returns `(result, our attack sent, CC2 attack sent)`.
fn run_versus(
    bin: &str,
    seed: u64,
    max_plies: u32,
    think: Duration,
) -> std::io::Result<(VersusResult, u32, u32)> {
    // A = our bot, on its own engine.
    let mut ours = VersusEngine::new(&|s| beam_linear_bot(s, 16, 2), seed);
    let mut ours_q = GarbageQueue::default();
    let mut ours_attack = 0u32;

    // B = CC2, mirrored in `sim`, fed the same seeded 7-bag as ours.
    let mut gen = PieceGenerator::with_seed(seed);
    let mut queue: VecDeque<PieceType> = (0..VISIBLE_QUEUE).map(|_| gen.next().unwrap()).collect();
    let mut hold: Option<PieceType> = None;
    let bag: Vec<String> = PieceType::all()
        .iter()
        .map(|&p| piece_letter(p).to_string())
        .collect();
    let q_letters = |q: &VecDeque<PieceType>| -> Vec<String> {
        q.iter().map(|&p| piece_letter(p).to_string()).collect()
    };
    let mut cc2 = Cc2::spawn(bin)?;
    cc2.start(&empty_tbp_board(), &q_letters(&queue), None, 0, false, &bag)?;
    let mut sim = Sim::new();
    let mut cc2_combo = 0u32;
    let mut cc2_b2b = false;
    let mut cc2_attack = 0u32;
    let mut cc2_q = GarbageQueue::default();

    // The referee's own hole stream (engine-rules matches draw holes inside each
    // receiver engine instead — see tetr-core's garbage module).
    let mut hole_rng = seed ^ tetr_research::versus_legacy::VERSUS_HOLE_SALT;
    let mut ours_topped = false;
    let mut cc2_topped = false;
    let mut cc2_plies = 0u32; // CC2 placements made
    let mut resyncs = 0u32; // stop+start re-syncs forced on CC2 by garbage

    'match_loop: for ply in 0..max_plies {
        // Alternate first mover so neither side gets a structural send-first edge.
        let order = if ply % 2 == 0 { [0u8, 1] } else { [1, 0] };
        for &who in &order {
            if who == 0 {
                // --- our ply ---
                let (atk, topped) = ours.step_piece();
                if topped {
                    ours_topped = true;
                    break 'match_loop;
                }
                let leftover = ours_q.cancel(atk);
                if leftover > 0 {
                    cc2_q.push(leftover, versus_hole(&mut hole_rng));
                    ours_attack += leftover;
                }
                for (lines, hcol) in ours_q.drain_newest_first() {
                    if ours.receive(lines, hcol) {
                        ours_topped = true;
                        break 'match_loop;
                    }
                }
            } else {
                // --- CC2 ply ---
                let Some(mv) = cc2.suggest(think)? else {
                    cc2_topped = true;
                    break 'match_loop;
                };
                cc2_plies += 1;
                let placed = piece_from_letter(&mv.location.piece);
                advance_queue_hold(&mut queue, &mut hold, placed, seed);

                let lines = sim.place_and_clear(&cc2_cells(&mv)) as usize;
                let atk = score_cc2_clear(&sim, &mv.spin, lines, &mut cc2_combo, &mut cc2_b2b);

                let leftover = cc2_q.cancel(atk);
                if leftover > 0 {
                    ours_q.push(leftover, versus_hole(&mut hole_rng));
                    cc2_attack += leftover;
                }

                let revealed = gen.next().unwrap();
                queue.push_back(revealed);
                cc2.play(&mv, piece_letter(revealed))?;

                // Dump remaining garbage onto CC2's board, then re-sync CC2 to it.
                let batches = cc2_q.drain_newest_first();
                if !batches.is_empty() {
                    let mut overflow = false;
                    for (lines, hcol) in batches {
                        overflow |= sim.insert_garbage(lines as usize, hcol);
                    }
                    if overflow {
                        cc2_topped = true;
                        break 'match_loop;
                    }
                    cc2.stop()?;
                    cc2.start(
                        &sim_to_tbp_board(&sim),
                        &q_letters(&queue),
                        hold.map(piece_letter),
                        cc2_combo,
                        cc2_b2b,
                        &bag,
                    )?;
                    resyncs += 1;
                }
            }
        }
    }

    let _ = cc2.stop();
    eprintln!(
        "    [diag seed {seed}] cc2_topped={cc2_topped} cc2_plies={cc2_plies} resyncs={resyncs} | ours_topped={ours_topped}",
    );
    // A = ours, B = CC2 (a topout loses before attack is compared — see `decide_versus`).
    let result = decide_versus(ours_topped, cc2_topped, ours_attack, cc2_attack);
    Ok((result, ours_attack, cc2_attack))
}

fn main() -> std::io::Result<()> {
    let bin = std::env::var("CC2_BIN")
        .unwrap_or_else(|_| "/tmp/cold-clear-2/target/release/cold-clear-2".to_string());
    let n_seeds = env_usize("SEEDS", 6);
    let pieces = env_usize("PIECES", 100);
    let think = Duration::from_millis(env_usize("THINK_MS", 50) as u64);

    eprintln!(
        "Cold Clear 2 baseline — {n_seeds} seeds × {pieces} pieces, think {think:?}/move, bin {bin}"
    );
    let seeds = seed_set(n_seeds);

    // Downstack comparison: head-to-head cheese-clear efficiency, our beam vs CC2.
    if std::env::var("DOWNSTACK").is_ok() {
        let garbage_rows = env_usize("GARBAGE_ROWS", 9) as u32;
        let cap = pieces as u32;
        let ours = evaluate_downstack(&|s| beam_linear_bot(s, 16, 2), &seeds, garbage_rows, cap);
        let mut cc2_sum = 0.0f64;
        let mut cc2_cleared = 0usize;
        for &seed in &seeds {
            let (p, cleared) = run_downstack(&bin, seed, garbage_rows, cap, think)?;
            eprintln!("  CC2 seed {seed:>20}: pieces={p:>3} cleared={cleared}");
            if cleared {
                cc2_sum += p as f64;
                cc2_cleared += 1;
            }
        }
        let cc2_mean = if cc2_cleared > 0 {
            cc2_sum / cc2_cleared as f64
        } else {
            0.0
        };
        println!(
            "downstack {garbage_rows} rows — pieces-to-clear (lower=better): OURS {:.2} ({:.0}% clear) | CC2 {:.2} ({}/{} clear)",
            ours.mean_pieces_to_clear,
            ours.clear_rate * 100.0,
            cc2_mean,
            cc2_cleared,
            seeds.len()
        );
        return Ok(());
    }

    // Versus head-to-head: our beam vs CC2 with mutual garbage.
    //
    // NOT A FAIR COMPARISON — kept for infrastructure, reported with a loud caveat.
    // Base TBP has no incremental-garbage message, so injecting garbage into CC2
    // forces a stop+start re-sync that discards its search tree. CC2 ends up
    // forfeiting while sending almost no attack (~4, vs its 0.69-APP baseline) —
    // it is crippled by the harness, not out-played. A credible versus result needs
    // a TBP garbage extension (if CC2 supports one) or a custom protocol. Until
    // then the rigorous, *fairly measurable* CC2 comparison is DOWNSTACK.
    if std::env::var("VERSUS").is_ok() {
        let max_plies = env_usize("MAX_PLIES", 60) as u32;
        let (mut ours_wins, mut cc2_wins, mut draws) = (0usize, 0usize, 0usize);
        let (mut ours_atk_sum, mut cc2_atk_sum) = (0u32, 0u32);
        for &seed in &seeds {
            let (res, ours_atk, cc2_atk) = run_versus(&bin, seed, max_plies, think)?;
            ours_atk_sum += ours_atk;
            cc2_atk_sum += cc2_atk;
            match res {
                VersusResult::AWins => ours_wins += 1,
                VersusResult::BWins => cc2_wins += 1,
                VersusResult::Draw => draws += 1,
            }
            eprintln!("  seed {seed:>20}: {res:?} | ours atk {ours_atk:>3} | cc2 atk {cc2_atk:>3}");
        }
        let n = seeds.len().max(1) as f64;
        println!("versus_ours_win_rate {:.2}", ours_wins as f64 / n);
        eprintln!(
            "  !! NOT FAIR: CC2 is crippled by TBP re-sync (see source); treat as infra only."
        );
        eprintln!(
            "VERSUS ours vs CC2 | OURS {ours_wins} / CC2 {cc2_wins} / draw {draws} | mean attack ours {:.1} cc2 {:.1} | {} seeds, {max_plies} plies, {think:?}/move",
            ours_atk_sum as f64 / n,
            cc2_atk_sum as f64 / n,
            seeds.len(),
        );
        return Ok(());
    }

    let mut total_app = 0.0f64;
    for &seed in &seeds {
        let attack = run_one(&bin, seed, pieces, think)?;
        let app = attack as f64 / pieces as f64;
        total_app += app;
        eprintln!("  seed {seed:>20}: attack={attack:>4}  APP={app:.4}");
    }
    let mean_app = total_app / n_seeds.max(1) as f64;
    println!("cc2_attack_per_piece {mean_app:.4}");
    Ok(())
}
