//! Terminal progress, one house style — cosmetic ONLY.
//!
//! Bars draw to stderr and auto-hide when stderr is not a TTY (CI, smoke,
//! piped logs see nothing). Machine lines stay on stdout and receipts are
//! untouched: nothing here may ever carry information that exists nowhere
//! else. Use [`ProgressBar::suspend`] around `eprintln!` while a bar is
//! live so log lines and the bar don't fight over the terminal.

use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// A bounded bar: `prefix [####----] pos/len msg`.
pub fn bar(len: u64, prefix: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{prefix:>10} [{bar:24}] {pos}/{len} {msg}")
            .expect("static template")
            .progress_chars("##-"),
    );
    pb.set_prefix(prefix.to_string());
    pb
}

/// An unbounded spinner: `prefix ⠋ msg` (for open-ended races).
pub fn spinner(prefix: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{prefix:>10} {spinner} {msg}").expect("static template"),
    );
    pb.set_prefix(prefix.to_string());
    pb.enable_steady_tick(Duration::from_millis(120));
    pb
}
