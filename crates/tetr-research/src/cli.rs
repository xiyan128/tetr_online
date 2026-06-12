//! Shared config + RNG helpers for the research `bin/` tools.
//!
//! Every bin is configured purely by environment variables — so a run is a function
//! of its env plus the engine seed, reproducible with no flags to thread — and the
//! hill-climbers need a dependency-free deterministic PRNG. Both were copy-pasted
//! into each bin (where the combo-bug-style desync risk lives); this is their single
//! home.

use std::any::type_name;
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, serde::Serialize)]
struct ResolvedEnv {
    raw: Option<String>,
    value: String,
    source: &'static str,
}

#[derive(Clone, Debug)]
struct RegistryEntry {
    default: String,
    resolved: ResolvedEnv,
}

static ENV_REGISTRY: OnceLock<Mutex<BTreeMap<String, RegistryEntry>>> = OnceLock::new();

fn registry() -> &'static Mutex<BTreeMap<String, RegistryEntry>> {
    ENV_REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn read_env(key: &str) -> Option<String> {
    std::env::var_os(key).map(|value| value.to_string_lossy().into_owned())
}

/// A set environment variable that could not be resolved as requested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    key: String,
    raw: String,
    expected: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}={:?} is not a valid {}",
            self.key, self.raw, self.expected
        )
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug)]
struct Resolution<T> {
    value: T,
    raw: Option<String>,
    rendered: String,
    source: &'static str,
}

fn resolve_with<T>(
    key: &str,
    raw: Option<String>,
    default: T,
    default_rendered: String,
    expected: impl Into<String>,
    parse: impl FnOnce(&str) -> Option<T>,
    render: impl Fn(&T) -> String,
) -> Result<Resolution<T>, ConfigError> {
    match raw {
        Some(raw) => {
            let value = parse(&raw).ok_or_else(|| ConfigError {
                key: key.to_string(),
                raw: raw.clone(),
                expected: expected.into(),
            })?;
            let rendered = render(&value);
            Ok(Resolution {
                value,
                raw: Some(raw),
                rendered,
                source: "env",
            })
        }
        None => Ok(Resolution {
            value: default,
            raw: None,
            rendered: default_rendered,
            source: "default",
        }),
    }
}

fn parse_or_default<T>(
    key: &str,
    raw: Option<String>,
    default: T,
) -> Result<Resolution<T>, ConfigError>
where
    T: FromStr + ToString,
{
    let default_rendered = default.to_string();
    resolve_with(
        key,
        raw,
        default,
        default_rendered,
        type_name::<T>(),
        |value| value.parse().ok(),
        ToString::to_string,
    )
}

fn record<T>(key: &str, default: String, resolution: Resolution<T>) -> T {
    let mut entries = registry().lock().expect("environment registry poisoned");
    if let Some(previous) = entries.get(key) {
        assert_eq!(
            previous.default, default,
            "config drift: {key} was read with defaults {:?} and {:?}",
            previous.default, default
        );
    } else {
        entries.insert(
            key.to_string(),
            RegistryEntry {
                default,
                resolved: ResolvedEnv {
                    raw: resolution.raw,
                    value: resolution.rendered,
                    source: resolution.source,
                },
            },
        );
    }
    resolution.value
}

fn or_exit<T>(result: Result<T, ConfigError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("config error: {error}");
            std::process::exit(2);
        }
    }
}

/// Parse environment variable `key` as `T`, using `default` only when it is unset.
/// A set value that does not parse is a configuration error and exits with status 2.
pub fn env_or<T>(key: &str, default: T) -> T
where
    T: FromStr + ToString,
{
    let default_rendered = default.to_string();
    let resolution = or_exit(parse_or_default(key, read_env(key), default));
    record(key, default_rendered, resolution)
}

/// [`env_or`] specialized to `usize` (the common case: seed counts, depths, widths).
pub fn env_usize(key: &str, default: usize) -> usize {
    env_or(key, default)
}

/// [`env_or`] specialized to `f64` (rates, sigmas, SPRT hypotheses).
pub fn env_f64(key: &str, default: f64) -> f64 {
    env_or(key, default)
}

/// Record whether `key` is present. The environment value itself is not interpreted.
pub fn env_flag(key: &str) -> bool {
    let raw = read_env(key);
    let value = raw.is_some();
    record(
        key,
        false.to_string(),
        Resolution {
            value,
            raw,
            rendered: value.to_string(),
            source: if value { "env" } else { "default" },
        },
    )
}

/// Resolve a string choice, rejecting set values outside `allowed` with status 2.
pub fn env_choice(key: &str, default: &str, allowed: &[&str]) -> String {
    assert!(
        allowed.contains(&default),
        "default {default:?} for {key} is not an allowed choice"
    );
    let expected = format!("choice (allowed: {})", allowed.join(", "));
    let resolution = or_exit(resolve_with(
        key,
        read_env(key),
        default.to_string(),
        default.to_string(),
        expected,
        |value| allowed.contains(&value).then(|| value.to_string()),
        Clone::clone,
    ));
    record(key, default.to_string(), resolution)
}

/// Resolve an arbitrary string value while recording its default and source.
pub fn env_string(key: &str, default: &str) -> String {
    let resolution = resolve_with(
        key,
        read_env(key),
        default.to_string(),
        default.to_string(),
        "string",
        |value| Some(value.to_string()),
        Clone::clone,
    )
    .expect("string parsing is infallible");
    record(key, default.to_string(), resolution)
}

/// Resolve exactly `N` comma-separated `f32` values, using `default` only when unset.
pub fn env_f32_array<const N: usize>(key: &str, default: [f32; N]) -> [f32; N] {
    let render = |values: &[f32; N]| {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    let default_rendered = render(&default);
    let resolution = or_exit(resolve_with(
        key,
        read_env(key),
        default,
        default_rendered.clone(),
        format!("comma-separated list of {N} f32 values"),
        |raw| {
            let values = raw
                .split(',')
                .map(str::trim)
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>()
                .ok()?;
            values.try_into().ok()
        },
        render,
    ));
    record(key, default_rendered, resolution)
}

/// Snapshot every environment value resolved through this module.
pub fn resolved_env() -> serde_json::Value {
    let entries = registry().lock().expect("environment registry poisoned");
    serde_json::to_value(
        entries
            .iter()
            .map(|(key, entry)| (key, &entry.resolved))
            .collect::<BTreeMap<_, _>>(),
    )
    .expect("resolved environment entries are serializable")
}

/// A tiny deterministic [SplitMix64](https://prng.di.unimi.it/splitmix64.c) PRNG —
/// the hill-climbers' mutation / jitter source. No `rand` dependency and fully
/// reproducible from the seed, so a climb replays bit-for-bit.
///
/// The seed is the running state and the standard increment is folded in on the
/// first [`next_u64`](Self::next_u64) — identical to the per-bin free-function form
/// it replaced (`SplitMix64::new(s).next_u64()` == old `next_u64(&mut s)`), so a
/// refactored climb produces the same sequence.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Wrap a bare running-state word as a generator — the inverse of
    /// [`into_raw`](Self::into_raw). Identical to [`new`](Self::new) (both store the
    /// word as the running state folded forward on the next [`next_u64`](Self::next_u64));
    /// the distinct name documents intent at call sites that thread a raw `u64` PRNG
    /// state through `&mut u64` rather than seeding a fresh stream.
    pub fn from_raw(state: u64) -> Self {
        Self(state)
    }

    /// Unwrap the running-state word — the inverse of [`from_raw`](Self::from_raw) — so
    /// a caller holding a bare `u64` can read the advanced state back after stepping.
    pub fn into_raw(self) -> u64 {
        self.0
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ok() {
        let resolved = parse_or_default("COUNT", Some("17".to_string()), 6usize).unwrap();
        assert_eq!(resolved.value, 17);
        assert_eq!(resolved.source, "env");
    }

    #[test]
    fn unset_uses_default() {
        let resolved = parse_or_default("COUNT", None, 6usize).unwrap();
        assert_eq!(resolved.value, 6);
        assert_eq!(resolved.source, "default");
    }

    #[test]
    fn parse_failure_names_key_and_raw_value() {
        let error = parse_or_default("COUNT", Some("many".to_string()), 6usize).unwrap_err();
        assert_eq!(error.to_string(), "COUNT=\"many\" is not a valid usize");
    }

    #[test]
    fn choice_rejects_unknown() {
        let error = resolve_with(
            "BOT",
            Some("tabu".to_string()),
            "beam".to_string(),
            "beam".to_string(),
            "choice (allowed: beam, bf)",
            |value| ["beam", "bf"].contains(&value).then(|| value.to_string()),
            Clone::clone,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "BOT=\"tabu\" is not a valid choice (allowed: beam, bf)"
        );
    }

    #[test]
    fn registry_records_resolved_sequence() {
        let key_default = "TETR_RESEARCH_TEST_DEFAULT";
        let key_env = "TETR_RESEARCH_TEST_ENV";
        let default = parse_or_default(key_default, None, 9usize).unwrap();
        record(key_default, "9".to_string(), default);
        let from_env = parse_or_default(key_env, Some("2.5".to_string()), 1.0f64).unwrap();
        record(key_env, "1".to_string(), from_env);

        let values = resolved_env();
        assert_eq!(values[key_default]["raw"], serde_json::Value::Null);
        assert_eq!(values[key_default]["value"], "9");
        assert_eq!(values[key_default]["source"], "default");
        assert_eq!(values[key_env]["raw"], "2.5");
        assert_eq!(values[key_env]["value"], "2.5");
        assert_eq!(values[key_env]["source"], "env");
    }
}
