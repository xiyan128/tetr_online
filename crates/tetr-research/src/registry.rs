//! THE experiment registry: every runnable configuration, by name, as code.
//!
//! This file is the crate's entire configuration surface. An experiment is a
//! named [`Entry`] holding a typed spec literal; the CLI runs entries by
//! name (`run <name>`), prints them (`show <name>`), and lists them
//! (`list`). Nothing else configures an experiment — no environment
//! variables, no config files, no per-knob flags.
//!
//! The discipline this buys: **a recorded result reproduces from
//! `(commit, name)`.** Changing anything that could change a result means
//! REGISTERING A NEW NAME (one literal, versioned with the code), never
//! mutating a name that has recorded runs — `resume` refuses a checkpoint
//! whose stored spec no longer matches its entry, and a dirty working tree
//! is stamped into every manifest, so even mid-edit experimentation stays
//! visible. Names are kebab-case and should say what the experiment IS, not
//! how it is tuned (`cc2-board-climb`, not `climb-margin150`): the tuning
//! lives in the literal below and in the manifest.
//!
//! A promotion is a configuration too: paste a climb's `best_params` into a
//! new `Promote` entry and run it by name — the candidate is then recorded
//! exactly like everything else.

use crate::commands::{Beam, ab, behavior, cc2_baseline, cc2_native, climb, metric, promote, sprt};

/// One runnable experiment: a name, a one-line description, and its spec.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: &'static str,
    pub about: &'static str,
    pub experiment: Experiment,
}

/// Every experiment kind the platform runs, with its typed spec. Serializes
/// internally tagged (`"kind": "climb", …fields`) — the form `show` prints
/// and `spec.json` records.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Experiment {
    Metric(metric::Spec),
    Marathon(marathon::Spec),
    Behavior(behavior::Spec),
    Ab(ab::Spec),
    Climb(climb::Spec),
    Race(sprt::Spec),
    Promote(promote::Spec),
    Cc2Native(cc2_native::Spec),
    Cc2Baseline(cc2_baseline::Spec),
}

use crate::commands::marathon;

fn e(name: &'static str, about: &'static str, experiment: Experiment) -> Entry {
    Entry {
        name,
        about,
        experiment,
    }
}

/// The catalog. Grouped by purpose; keep entries that have recorded runs.
pub fn entries() -> Vec<Entry> {
    use Experiment::*;
    vec![
        // --- iteration metrics (the /autoresearch parse contracts) --------
        e(
            "app-metric",
            "capped-marathon score/sec + APP, the tight-loop headline",
            Metric(metric::Spec::marathon()),
        ),
        e(
            "downstack-metric",
            "censored cheese pieces-to-clear + clear rate (lower = better)",
            Metric(metric::Spec::downstack()),
        ),
        e(
            "versus-metric",
            "quick win rate vs the greedy baseline",
            Metric(metric::Spec::versus()),
        ),
        // --- benchmarks and suites ----------------------------------------
        e(
            "marathon-sweep",
            "greedy vs beam depths 1-3 on score/sec (depth-1 must tie greedy)",
            Marathon(marathon::Spec::default()),
        ),
        e(
            "behavior-dt20",
            "APP / DS-P behavior suite for the shipped DT-20 beam",
            Behavior(behavior::Spec::bot(behavior::Bot::Dt20)),
        ),
        e(
            "behavior-cc2",
            "APP / DS-P behavior suite for the ported CC2 evaluator",
            Behavior(behavior::Spec::bot(behavior::Bot::Cc2)),
        ),
        e(
            "awareness-ab",
            "garbage-awareness A/B: CC2 beam vs its blinded twin, arm-swapped",
            Ab(ab::Spec::bot(ab::Bot::Beam)),
        ),
        e(
            "awareness-ab-bf",
            "garbage-awareness A/B on the best-first arm (nodes 192, depth 6)",
            Ab(ab::Spec::bot(ab::Bot::Bf)),
        ),
        // --- versus science -------------------------------------------------
        e(
            "cc2-board-climb",
            "the production (1+1)-ES board-param climb: v3-calibrated screen \
             (margin 150, sigma 0.08), confirm α=0.02, anchored — campaign cc2-board-v4",
            Climb(climb::Spec {
                campaign: "cc2-board-v4".to_string(),
                accept_margin: 150.0,
                sigma: 0.08,
                ..climb::Spec::default()
            }),
        ),
        e(
            "race-v3-candidate",
            "the historical v3 judge (H0 ACCEPTED — see the climb run records)",
            Race(sprt::Spec::default()),
        ),
        e(
            "promote-null-check",
            "panel harness check: the origin as candidate must come back REJECT",
            Promote(promote::Spec::default()),
        ),
        // --- external baselines ----------------------------------------------
        e(
            "cc2-native-baseline",
            "CC2's ported evaluator vs DT-20 on our engine — the fair comparison",
            Cc2Native(cc2_native::Spec::default()),
        ),
        e(
            "cc2-baseline-app",
            "real CC2 over TBP: attack-per-piece on our seeded bag (--cc2-bin)",
            Cc2Baseline(cc2_baseline::Spec::mode(cc2_baseline::Mode::App)),
        ),
        e(
            "cc2-baseline-downstack",
            "real CC2 over TBP: refereed cheese race vs our beam (--cc2-bin)",
            Cc2Baseline(cc2_baseline::Spec::mode(cc2_baseline::Mode::Downstack)),
        ),
        // --- smoke (research-smoke + fast checks; deliberately tiny) --------
        e(
            "smoke-climb",
            "tiny anchored climb for the smoke gate (seconds; campaign smoke)",
            Climb(climb::Spec {
                campaign: "smoke".to_string(),
                beam: Beam { width: 4, depth: 1 },
                format: crate::versus::VersusFormat {
                    max_plies: 16,
                    rain_period: 4,
                },
                screen_seeds: 2,
                val_seeds: 2,
                accept_margin: 0.0,
                confirm_matches: 0,
                anchor_every: 2,
                anchor_matches: 0,
                ..climb::Spec::default()
            }),
        ),
        e(
            "smoke-promote",
            "tiny promotion panel for the smoke gate (null check at toy sizes)",
            Promote(promote::Spec {
                campaign: "smoke".to_string(),
                cell_matches: 4,
                max_plies: 16,
                beam: Beam { width: 4, depth: 1 },
                ..promote::Spec::default()
            }),
        ),
        e(
            "smoke-metric",
            "downstack metric at 4 seeds (the parse-contract canary)",
            Metric(metric::Spec {
                seeds: 4,
                ..metric::Spec::downstack()
            }),
        ),
    ]
}

/// Look a registered experiment up by name.
pub fn find(name: &str) -> Option<Entry> {
    entries().into_iter().find(|e| e.name == name)
}

/// The spec exactly as `show` prints it and `spec.json` records it.
pub fn spec_json(experiment: &Experiment) -> serde_json::Value {
    serde_json::to_value(experiment).expect("registry specs serialize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_kebab() {
        let all = entries();
        let mut names: Vec<_> = all.iter().map(|e| e.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate registry names");
        for name in names {
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                "registry name {name:?} is not kebab-case"
            );
        }
    }

    /// Every entry must serialize (it IS the manifest format) and round-trip
    /// through `find`.
    #[test]
    fn every_entry_shows_and_resolves() {
        for entry in entries() {
            let v = spec_json(&entry.experiment);
            assert!(
                v.get("kind").is_some(),
                "{}: spec lost its kind tag",
                entry.name
            );
            assert_eq!(find(entry.name).unwrap().name, entry.name);
        }
    }

    /// The smoke gate and docs reference these by name; renaming them breaks
    /// scripts silently — fail here instead.
    #[test]
    fn load_bearing_names_exist() {
        for name in [
            "smoke-climb",
            "smoke-promote",
            "smoke-metric",
            "downstack-metric",
            "app-metric",
            "cc2-board-climb",
            "promote-null-check",
        ] {
            assert!(find(name).is_some(), "missing load-bearing entry {name}");
        }
    }
}
