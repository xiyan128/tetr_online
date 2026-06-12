//! Machine-readable manifests for reproducible research runs.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Map, Value, json};

const SCHEMA_VERSION: u64 = 1;

/// A run directory containing its specification, per-seed outcomes, and summary.
pub struct RunLedger {
    dir: PathBuf,
    outcomes: File,
}

impl RunLedger {
    /// Create `<runs-root>/<YYYYMMDD-HHMMSS>-<bin>-<pid>/` and write `spec.json`.
    pub fn create(bin: &str, extra_spec: Value) -> io::Result<RunLedger> {
        Self::create_at(&runs_root(), bin, extra_spec)
    }

    /// Create a run ledger under an explicit root directory.
    pub fn create_at(root: &Path, bin: &str, extra_spec: Value) -> io::Result<RunLedger> {
        validate_bin(bin)?;
        let now = SystemTime::now();
        let run_id = format!("{}-{bin}-{}", compact_utc(now)?, std::process::id());
        let dir = root.join(&run_id);
        fs::create_dir_all(root)?;
        fs::create_dir(&dir)?;

        let git = git_metadata();
        let spec = json!({
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "bin": bin,
            "created_utc": rfc3339_utc(now)?,
            "git": git,
            "host": {
                "hostname": hostname(),
                "cores": std::thread::available_parallelism().map(usize::from).unwrap_or(1),
                "os": std::env::consts::OS,
            },
            "extra": extra_spec,
        });
        write_json(&dir.join("spec.json"), &spec)?;

        let outcomes = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(dir.join("outcomes.jsonl"))?;
        Ok(Self { dir, outcomes })
    }

    /// Append one JSON object as one line of `outcomes.jsonl`.
    pub fn append_outcome(&mut self, outcome: &impl Serialize) -> io::Result<()> {
        serde_json::to_writer(&mut self.outcomes, outcome).map_err(io::Error::other)?;
        self.outcomes.write_all(b"\n")?;
        self.outcomes.flush()
    }

    /// Write `summary.json`, adding the mandatory completion metadata.
    pub fn write_summary(&self, summary: Value) -> io::Result<()> {
        let mut fields = match summary {
            Value::Object(fields) => fields,
            other => {
                let mut fields = Map::new();
                fields.insert("result".to_string(), other);
                fields
            }
        };
        fields.insert("schema_version".to_string(), SCHEMA_VERSION.into());
        fields.insert(
            "finished_utc".to_string(),
            rfc3339_utc(SystemTime::now())?.into(),
        );
        fields
            .entry("exit_reason".to_string())
            .or_insert_with(|| "complete".into());
        write_json(&self.dir.join("summary.json"), &Value::Object(fields))
    }

    /// Atomically write or replace `checkpoint.json` within the run directory.
    pub fn write_checkpoint(&self, state: Value) -> io::Result<()> {
        let tmp = self
            .dir
            .join(format!("checkpoint.json.tmp-{}", std::process::id()));
        write_json(&tmp, &state)?;
        fs::rename(tmp, self.dir.join("checkpoint.json"))
    }

    /// Read `checkpoint.json` from an existing run directory.
    pub fn read_checkpoint(run_dir: &Path) -> io::Result<Value> {
        let file = File::open(run_dir.join("checkpoint.json"))?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }

    /// Path to this run's directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// The default run-directory root: `<git-toplevel>/runs`, falling back to
/// `./runs` outside git.
pub fn runs_root() -> PathBuf {
    git_output(&["rev-parse", "--show-toplevel"])
        .map(|top| PathBuf::from(top).join("runs"))
        .unwrap_or_else(|| PathBuf::from("runs"))
}

fn validate_bin(bin: &str) -> io::Result<()> {
    if !bin.is_empty()
        && bin
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid ledger bin name {bin:?}"),
        ))
    }
}

fn write_json(path: &Path, value: &Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
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

fn hostname() -> String {
    Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|hostname| !hostname.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
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
    fn tempdir_round_trip() {
        let root = std::env::temp_dir().join(format!(
            "tetr-research-ledger-test-{}-{}",
            std::process::id(),
            compact_utc(SystemTime::now()).unwrap()
        ));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }

        let mut ledger =
            RunLedger::create_at(&root, "ledger-test", json!({"format": "test"})).unwrap();
        ledger.append_outcome(&json!({"seed": 1})).unwrap();
        ledger.append_outcome(&json!({"seed": 2})).unwrap();
        ledger.append_outcome(&json!({"seed": 3})).unwrap();
        ledger.write_checkpoint(json!({"step": 1})).unwrap();
        ledger.write_checkpoint(json!({"step": 2})).unwrap();
        ledger
            .write_summary(json!({"exit_reason": "complete", "games": 3}))
            .unwrap();

        let dir = ledger.dir().to_path_buf();
        assert!(dir.join("spec.json").is_file());
        assert!(dir.join("outcomes.jsonl").is_file());
        assert!(dir.join("checkpoint.json").is_file());
        assert!(dir.join("summary.json").is_file());
        let outcomes = fs::read_to_string(dir.join("outcomes.jsonl")).unwrap();
        assert_eq!(outcomes.lines().count(), 3);
        assert_eq!(RunLedger::read_checkpoint(&dir).unwrap()["step"], 2);
        let spec: Value =
            serde_json::from_reader(File::open(dir.join("spec.json")).unwrap()).unwrap();
        assert_eq!(spec["extra"]["format"], "test");

        drop(ledger);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn epoch_formats_as_utc() {
        assert_eq!(rfc3339_utc(UNIX_EPOCH).unwrap(), "1970-01-01T00:00:00Z");
        assert_eq!(compact_utc(UNIX_EPOCH).unwrap(), "19700101-000000");
    }
}
