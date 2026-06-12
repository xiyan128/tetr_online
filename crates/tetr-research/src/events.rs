//! The game stream: raw per-game facts, fully normalized — duckdb does the
//! denormalizing on read, the platform never reads these back.
//!
//! First principles: **a game is the only fact**, and nothing is stored that
//! the receipt or the file itself already determines. No timestamps, no run
//! id (the path), no kind/mode (the receipt's eval), no bot names (the
//! receipt's `bots` plus a per-game `swapped` bit for arm-swapped pairings),
//! no derived results (queries). The one ordinal exception is `n`: row order
//! is semantic (a race's LLR is a fold over its ordered games) and duckdb
//! cannot recover line numbers from JSON, so each row carries its index.
//!
//! Consequence worth its weight: with nothing observational in the rows,
//! `games.jsonl` is BYTE-IDENTICAL across replays of the same run — `diff`
//! is a replay witness (the smoke asserts it). Scope: that holds for
//! fixed-work runs (every eval). A budget-truncated optimizer run stops at a
//! wall-clock-dependent iteration, so two invocations reproduce as a PREFIX
//! relation, byte-identical only up to the shorter stream (the walk itself
//! is deterministic; only the stopping point is machine-local).
//!
//! Emission is a no-op until the RUNNER installs the sink (commands stay
//! tracking-blind; tests and library use stay silent), never fails a run,
//! and happens AFTER order-stable collection. Analysis: `duckdb -init
//! scripts/research.sql`; streaming: `tail -f runs/<id>/games.jsonl`.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

static SINK: OnceLock<Mutex<File>> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

/// Install the run's sink (`<run-dir>/games.jsonl`). Once per process, by
/// the runner, after the receipt exists.
pub fn install(run_dir: &Path) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(run_dir.join("games.jsonl"))?;
    let _ = SINK.set(Mutex::new(file));
    Ok(())
}

/// Emit one game's facts. No sink ⇒ no-op; write failure ⇒ swallowed
/// (events observe runs, they must never end one).
pub fn game(payload: Value) {
    let Some(sink) = SINK.get() else { return };
    let mut row = json!({ "n": SEQ.fetch_add(1, Ordering::Relaxed) });
    if let (Value::Object(row), Value::Object(payload)) = (&mut row, payload) {
        row.extend(payload);
    }
    if let Ok(mut file) = sink.lock() {
        let _ = serde_json::to_writer(&mut *file, &row);
        let _ = file.write_all(b"\n");
    }
}

/// Seeds travel as hex strings: u64s corrupt through f64-only JSON readers,
/// and a seed is an identifier, not arithmetic.
pub fn seed_hex(seed: u64) -> String {
    format!("{seed:#018x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unsinked emission must be free and silent — the library path.
    #[test]
    fn emit_without_a_sink_is_a_noop() {
        game(json!({"seed": seed_hex(7)}));
    }

    #[test]
    fn seeds_render_as_full_width_hex() {
        assert_eq!(seed_hex(0xdead_beef), "0x00000000deadbeef");
    }
}
