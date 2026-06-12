//! Inspect the run ledger: every recorded run with its experiment, creation
//! time, and exit reason — the index into `runs/` an agent reads before
//! resuming or citing anything.

use std::fs::File;
use std::path::Path;

use serde_json::Value;

use crate::ledger;

fn field<'v>(v: &'v Value, path: &[&str]) -> Option<&'v str> {
    let mut cur = v;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_str()
}

/// Print the most recent `last` runs under `root` (default ledger root),
/// oldest first so the freshest line is at the prompt.
pub fn list(root: Option<&Path>, last: usize) -> std::io::Result<()> {
    let root = root.map_or_else(ledger::runs_root, Path::to_path_buf);
    let mut dirs: Vec<_> = match std::fs::read_dir(&root) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(_) => {
            println!("no runs at {}", root.display());
            return Ok(());
        }
    };
    // Run ids start with a UTC timestamp, so name order is time order.
    dirs.sort();
    let skip = dirs.len().saturating_sub(last);
    for dir in &dirs[skip..] {
        let spec: Option<Value> = File::open(dir.join("spec.json"))
            .ok()
            .and_then(|f| serde_json::from_reader(f).ok());
        let experiment = spec
            .as_ref()
            .and_then(|s| field(s, &["experiment"]))
            .unwrap_or("?");
        let created = spec
            .as_ref()
            .and_then(|s| field(s, &["created_utc"]))
            .unwrap_or("?");
        let resumable = dir.join("checkpoint.json").exists();
        println!(
            "{created}  {experiment:<24}{} {}",
            if resumable { " [checkpoint]" } else { "" },
            dir.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        );
    }
    Ok(())
}
