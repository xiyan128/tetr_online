//! Native Cold Clear 2 head-to-head: CC2's **ported** evaluator
//! ([`tetr_core::ai::Cc2Evaluator`]) vs our DT-20 evaluator, both on the SAME beam,
//! engine, and garbage rules. Fair by construction — no TBP, no re-sync, both bots
//! play real mutual garbage on our engine. This is the comparison the TBP bridge
//! could not give (CC2 has no garbage message), and the baseline we hillclimb past.
//!
//! Env: `SEEDS` (12), `BEAM_DEPTH` (2), `BEAM_WIDTH` (16), `MAX_PLIES` (160),
//!      `GARBAGE_ROWS` (9), `MAX_PIECES` (100).

use tetr_research::{beam_cc2_bot, beam_linear_bot, evaluate_downstack, evaluate_versus, seed_set};

use tetr_research::cli::env_usize;

fn main() {
    let seeds = seed_set(env_usize("SEEDS", 12));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let plies = env_usize("MAX_PLIES", 160) as u32;
    let rows = env_usize("GARBAGE_ROWS", 9) as u32;
    let cap = env_usize("MAX_PIECES", 100) as u32;

    eprintln!(
        "Native CC2-eval vs DT-20-eval — both on beam(depth={depth}, width={width}); {} seeds",
        seeds.len()
    );

    // --- Versus (fair, our engine, mutual garbage): A = CC2-eval, B = DT-20 ---
    let vs = evaluate_versus(
        &|s| beam_cc2_bot(s, width, depth),
        &|s| beam_linear_bot(s, width, depth),
        &seeds,
        plies,
    );
    println!("versus_cc2eval_win_rate {:.2}", vs.a_win_rate());
    eprintln!(
        "VERSUS  CC2-eval(A) vs DT20(B) | CC2 {} / DT20 {} / draw {} | mean attack CC2 {:.1} DT20 {:.1} | {plies} plies",
        vs.a_wins, vs.b_wins, vs.draws, vs.mean_attack_a, vs.mean_attack_b
    );

    // --- Downstack: defense (pieces, lower=better) + offense (attack, higher) ---
    let cc2_ds = evaluate_downstack(&|s| beam_cc2_bot(s, width, depth), &seeds, rows, cap);
    let dt_ds = evaluate_downstack(&|s| beam_linear_bot(s, width, depth), &seeds, rows, cap);
    eprintln!(
        "DOWNSTACK {rows} rows | CC2-eval: {:.2} pieces  {:.1} attack  {:.0}% clear  ||  DT20: {:.2} pieces  {:.1} attack  {:.0}% clear",
        cc2_ds.mean_pieces_to_clear,
        cc2_ds.mean_attack,
        cc2_ds.clear_rate * 100.0,
        dt_ds.mean_pieces_to_clear,
        dt_ds.mean_attack,
        dt_ds.clear_rate * 100.0,
    );
    println!("downstack_cc2_pieces {:.2}", cc2_ds.mean_pieces_to_clear);
    println!("downstack_dt20_pieces {:.2}", dt_ds.mean_pieces_to_clear);
}
