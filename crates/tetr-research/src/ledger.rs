//! Run receipts and resume checkpoints — the ONLY persistence in the crate.
//!
//! Tracking is deliberately not a participant in experiments: commands never
//! see this module. The runner writes one `spec.json` RECEIPT per run (the
//! reproducibility coordinates: experiment name, typed spec, runtime, git
//! state) before dispatch. Anything richer — metrics sinks,
//! wandb-style dashboards — belongs in an observer that reads receipts and
//! the commands' stdout machine lines, not in here.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

const SCHEMA_VERSION: u64 = 2;

/// One run's directory, holding its receipt.
pub struct RunDir {
    dir: PathBuf,
}

impl RunDir {
    /// Create `<root>/<YYYYMMDD-HHMMSS>-<experiment>-<pid>/` (root defaults
    /// to [`runs_root`]) and write the `spec.json` receipt.
    pub fn create(root: Option<&Path>, experiment: &str, receipt: Value) -> io::Result<RunDir> {
        validate_name(experiment)?;
        let root = root.map_or_else(runs_root, Path::to_path_buf);
        let now = SystemTime::now();
        let run_id = format!("{}-{experiment}-{}", compact_utc(now)?, std::process::id());
        let dir = root.join(&run_id);
        fs::create_dir_all(&root)?;
        fs::create_dir(&dir)?;

        // The run id is the directory name — stored nowhere else.
        let mut spec = json!({
            "schema_version": SCHEMA_VERSION,
            "created_utc": rfc3339_utc(now)?,
            "git": git_metadata(),
        });
        if let (Value::Object(spec), Value::Object(receipt)) = (&mut spec, receipt) {
            spec.extend(receipt);
        }
        write_json(&dir.join("spec.json"), &spec)?;
        Ok(Self { dir })
    }

    /// Path to this run's directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Working-tree dirtiness: `Some(true)` = uncommitted changes, `None` = not
/// a git checkout. Both mean a run is not re-runnable from
/// `(commit, eval, bots…)` — the runner gates on this unless bypassed.
pub fn dirty() -> Option<bool> {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .and_then(|output| output.status.success().then_some(!output.stdout.is_empty()))
}

/// The default run-directory root: `<git-toplevel>/runs`, falling back to
/// `./runs` outside git.
pub fn runs_root() -> PathBuf {
    git_output(&["rev-parse", "--show-toplevel"])
        .map(|top| PathBuf::from(top).join("runs"))
        .unwrap_or_else(|| PathBuf::from("runs"))
}

fn validate_name(name: &str) -> io::Result<()> {
    if !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid run name {name:?}"),
        ))
    }
}

fn write_json(path: &Path, value: &Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value).map_err(io::Error::other)?;
    io::Write::write_all(&mut file, b"\n")?;
    file.sync_all()
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_metadata() -> Value {
    let commit = git_output(&["rev-parse", "HEAD"]);
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .and_then(|output| output.status.success().then_some(!output.stdout.is_empty()));
    json!({ "commit": commit, "dirty": dirty })
}

fn compact_utc(time: SystemTime) -> io::Result<String> {
    let (year, month, day, hour, minute, second) = utc_parts(time)?;
    Ok(format!(
        "{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}"
    ))
}

fn rfc3339_utc(time: SystemTime) -> io::Result<String> {
    let (year, month, day, hour, minute, second) = utc_parts(time)?;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

fn utc_parts(time: SystemTime) -> io::Result<(i64, i64, i64, i64, i64, i64)> {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
        .as_secs() as i64;
    let days = seconds / 86_400;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    Ok((
        year,
        month,
        day,
        seconds_of_day / 3_600,
        seconds_of_day % 3_600 / 60,
        seconds_of_day % 60,
    ))
}

// Gregorian calendar conversion from a Unix-epoch day count.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_round_trips() {
        let root = std::env::temp_dir().join(format!(
            "tetr-research-ledger-test-{}-{}",
            std::process::id(),
            compact_utc(SystemTime::now()).unwrap()
        ));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }

        let run = RunDir::create(
            Some(&root),
            "ledger-test",
            json!({"experiment": "ledger-test", "spec": {"kind": "test"}}),
        )
        .unwrap();

        let dir = run.dir().to_path_buf();
        assert!(dir.join("spec.json").is_file());
        let spec: Value =
            serde_json::from_reader(File::open(dir.join("spec.json")).unwrap()).unwrap();
        assert!(spec.get("run_id").is_none());
        assert_eq!(spec["experiment"], "ledger-test");
        assert_eq!(spec["spec"]["kind"], "test");
        assert!(spec["git"].is_object());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn epoch_formats_as_utc() {
        assert_eq!(rfc3339_utc(UNIX_EPOCH).unwrap(), "1970-01-01T00:00:00Z");
        assert_eq!(compact_utc(UNIX_EPOCH).unwrap(), "19700101-000000");
    }
}
