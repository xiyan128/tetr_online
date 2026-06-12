//! The event stream: every game the platform plays, one JSONL row, analyzed
//! by duckdb — the platform itself never reads these back.
//!
//! First principles: **a game is the only fact.** Parameters live in the
//! receipt (`spec.json`), judgments are queries over games × receipts, and a
//! run's headline lands as one `result` event for convenience. There are no
//! process events — even a race's LLR trajectory is a fold over its ordered
//! game stream, so storing it would be storing a computation.
//!
//! Two kinds:
//!
//! - `game` — raw outcomes of one match (solo: pieces/lines/score/attack;
//!   versus: per-side topped/attack), with the bot NAMES and the seed as a
//!   hex string (u64 seeds corrupt through f64-only JSON readers; hex is an
//!   identifier, not arithmetic).
//! - `result` — the run's machine lines as one object, plus `exit_reason`.
//!
//! The envelope stamps `ts_ms`, `run`, monotone `seq`, and `v`. Emission is
//! a no-op until the RUNNER installs a sink (commands stay tracking-blind;
//! tests and library use stay silent), events never fail a run (write errors
//! are swallowed), and emit AFTER order-stable collection — the file is then
//! deterministic modulo `ts_ms`, which doubles as a replay witness:
//!
//! ```text
//! duckdb -init scripts/research.sql            # views: runs/events/games/results
//! tail -f runs/<id>/events.jsonl | jq .        # stream a live run
//! ```

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

static SINK: OnceLock<Mutex<File>> = OnceLock::new();
static RUN: OnceLock<String> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

const SCHEMA_VERSION: u64 = 1;

/// Install the run's sink (`<run-dir>/events.jsonl`). Once per process, by
/// the runner, after the receipt exists.
pub fn install(run_id: &str, run_dir: &Path) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(run_dir.join("events.jsonl"))?;
    let _ = RUN.set(run_id.to_string());
    let _ = SINK.set(Mutex::new(file));
    Ok(())
}

/// Emit one event. No sink installed ⇒ no-op; write failure ⇒ swallowed
/// (events observe runs, they must never end one).
pub fn emit(kind: &str, payload: Value) {
    let Some(sink) = SINK.get() else { return };
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut event = json!({
        "v": SCHEMA_VERSION,
        "ts_ms": ts_ms,
        "run": RUN.get().map(String::as_str).unwrap_or(""),
        "seq": SEQ.fetch_add(1, Ordering::Relaxed),
        "kind": kind,
    });
    if let (Value::Object(event), Value::Object(payload)) = (&mut event, payload) {
        event.extend(payload);
    }
    if let Ok(mut file) = sink.lock() {
        let _ = serde_json::to_writer(&mut *file, &event);
        let _ = file.write_all(b"\n");
    }
}

/// Seeds travel as hex strings (see the module docs).
pub fn seed_hex(seed: u64) -> String {
    format!("{seed:#018x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unsinked emission must be free and silent — the library path.
    #[test]
    fn emit_without_a_sink_is_a_noop() {
        emit("game", json!({"seed": seed_hex(7)}));
    }

    #[test]
    fn seeds_render_as_full_width_hex() {
        assert_eq!(seed_hex(0xdead_beef), "0x00000000deadbeef");
    }
}
