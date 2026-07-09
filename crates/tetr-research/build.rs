//! Embed the BUILD-time commit so run receipts can distinguish the binary's
//! code from the source tree's state at run time. A stale binary silently
//! invalidated a night of anchor measurements on 2026-07-09 (the runtime
//! `git rev-parse` in spec.json reported the TREE's commit, not the code
//! that actually ran).

use std::process::Command;

fn main() {
    let out = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = out(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = out(&["status", "--porcelain"]).map(|s| !s.is_empty());
    println!("cargo:rustc-env=TETR_BUILD_COMMIT={commit}");
    println!(
        "cargo:rustc-env=TETR_BUILD_DIRTY={}",
        dirty
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
    // Re-run when HEAD moves (commit/checkout); dirty-flag staleness within a
    // HEAD is acceptable — the commit is the load-bearing field.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
