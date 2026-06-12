//! THE experiment registry: every runnable configuration, by name, as code.
//!
//! An entry BINDS an eval spec to named bots from [`crate::bots`] (or, for
//! the climb, names the subject it mutates). The CLI runs entries by name;
//! nothing else configures an experiment — no environment variables, no
//! config files, no per-knob flags.
//!
//! The discipline this buys: **a recorded result reproduces from
//! `(commit, name)`.** Changing anything that could change a result means
//! REGISTERING A NEW NAME (one literal, versioned with the code), never
//! mutating a name that has recorded runs — `resume` refuses a checkpoint
//! whose stored spec no longer matches its entry, and a dirty working tree
//! is stamped into every receipt.
//!
//! The bindings are deliberately thin: a climbed candidate gets ONE bot
//! registration and is then raceable, panelable, and benchmarkable here in
//! three one-line entries.

use crate::commands::{
    awareness, behavior, cc2_baseline, climb, downstack, marathon, panel, race, versus,
};

/// One runnable experiment: a name, a one-line description, and its binding.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: &'static str,
    pub about: &'static str,
    pub experiment: Experiment,
}

/// An eval spec bound to named bots (or the climb and its subject).
/// Serializes internally tagged — the form `show` prints and the receipt
/// records.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Experiment {
    Marathon {
        spec: marathon::Spec,
        bot: String,
    },
    Downstack {
        spec: downstack::Spec,
        bot: String,
    },
    Versus {
        spec: versus::Spec,
        a: String,
        b: String,
    },
    Behavior {
        spec: behavior::Spec,
        bot: String,
    },
    Awareness {
        spec: awareness::Spec,
        bot: String,
    },
    Race {
        spec: race::Spec,
        candidate: String,
        incumbent: String,
    },
    Panel {
        spec: panel::Spec,
        candidate: String,
    },
    Climb {
        spec: climb::Spec,
    },
    Cc2Baseline {
        spec: cc2_baseline::Spec,
    },
}

fn e(name: &'static str, about: &'static str, experiment: Experiment) -> Entry {
    Entry {
        name,
        about,
        experiment,
    }
}

fn s(name: &str) -> String {
    name.to_string()
}

/// The catalog. Grouped by purpose; names with recorded runs are permanent.
pub fn entries() -> Vec<Entry> {
    use Experiment::*;
    vec![
        // --- iteration metrics (the /autoresearch parse contracts) --------
        e(
            "app-metric",
            "capped-marathon score/sec + APP for the dt20 beam (tight loops)",
            Marathon {
                spec: marathon::Spec::default(),
                bot: s("dt20"),
            },
        ),
        e(
            "downstack-metric",
            "censored cheese pieces + clear rate for the dt20 beam",
            Downstack {
                spec: downstack::Spec::default(),
                bot: s("dt20"),
            },
        ),
        e(
            "versus-metric",
            "quick win rate: dt20 beam vs the greedy baseline",
            Versus {
                spec: versus::Spec {
                    seeds: 6,
                    format: crate::versus::VersusFormat {
                        max_plies: 120,
                        rain_period: 0,
                    },
                },
                a: s("dt20"),
                b: s("greedy"),
            },
        ),
        // --- suites ---------------------------------------------------------
        e(
            "behavior-dt20",
            "APP / DS-P behavior suite for the shipped DT-20 beam",
            Behavior {
                spec: behavior::Spec::default(),
                bot: s("dt20"),
            },
        ),
        e(
            "behavior-cc2",
            "APP / DS-P behavior suite for the ported CC2 evaluator",
            Behavior {
                spec: behavior::Spec::default(),
                bot: s("cc2-default"),
            },
        ),
        e(
            "awareness-ab",
            "garbage-awareness A/B: CC2 beam vs its blinded twin, arm-swapped",
            Awareness {
                spec: awareness::Spec::default(),
                bot: s("cc2-default"),
            },
        ),
        e(
            "awareness-ab-bf",
            "garbage-awareness A/B on the best-first arm (nodes 192)",
            Awareness {
                spec: awareness::Spec::default(),
                bot: s("bf-192"),
            },
        ),
        // --- versus science ---------------------------------------------------
        e(
            "cc2-board-climb",
            "the production (1+1)-ES board-param climb from attack-tuned: \
             v3-calibrated screen (margin 150, sigma 0.08), confirm α=0.02, \
             anchored — campaign cc2-board-v4",
            Climb {
                spec: climb::Spec {
                    campaign: s("cc2-board-v4"),
                    accept_margin: 150.0,
                    sigma: 0.08,
                    ..climb::Spec::default()
                },
            },
        ),
        e(
            "race-v3-candidate",
            "the historical v3 judge (H0 ACCEPTED — see the race run record)",
            Race {
                spec: race::Spec::default(),
                candidate: s("v3-candidate"),
                incumbent: s("attack-tuned"),
            },
        ),
        e(
            "panel-null-check",
            "panel harness check: attack-tuned as its own candidate must REJECT",
            Panel {
                spec: panel::Spec::default(),
                candidate: s("attack-tuned"),
            },
        ),
        // --- external baselines ------------------------------------------------
        e(
            "cc2-native-versus",
            "CC2's ported evaluator vs DT-20 on our engine — the fair comparison",
            Versus {
                spec: versus::Spec::default(),
                a: s("cc2-default"),
                b: s("dt20"),
            },
        ),
        e(
            "downstack-cc2eval",
            "censored cheese race, CC2's ported evaluator (vs downstack-metric)",
            Downstack {
                spec: downstack::Spec {
                    seeds: 12,
                    ..downstack::Spec::default()
                },
                bot: s("cc2-default"),
            },
        ),
        e(
            "cc2-baseline-app",
            "real CC2 over TBP: attack-per-piece on our seeded bag (--cc2-bin)",
            Cc2Baseline {
                spec: cc2_baseline::Spec::mode(cc2_baseline::Mode::App),
            },
        ),
        e(
            "cc2-baseline-downstack",
            "real CC2 over TBP: refereed cheese race vs our beam (--cc2-bin)",
            Cc2Baseline {
                spec: cc2_baseline::Spec::mode(cc2_baseline::Mode::Downstack),
            },
        ),
        // --- smoke (the research-smoke gate; deliberately tiny) ----------------
        e(
            "smoke-climb",
            "tiny anchored climb for the smoke gate (seconds; campaign smoke)",
            Climb {
                spec: climb::Spec {
                    campaign: s("smoke"),
                    subject: s("attack-tuned-tiny"),
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
                },
            },
        ),
        e(
            "smoke-panel",
            "tiny promotion panel for the smoke gate (starved cells must REJECT)",
            Panel {
                spec: panel::Spec {
                    campaign: s("smoke"),
                    must_beat: vec![s("greedy")],
                    must_not_lose_to: vec![s("attack-tuned-tiny")],
                    cell_matches: 4,
                    max_plies: 16,
                    ..panel::Spec::default()
                },
                candidate: s("attack-tuned-tiny"),
            },
        ),
        e(
            "smoke-downstack",
            "downstack metric at 4 seeds (the parse-contract canary)",
            Downstack {
                spec: downstack::Spec {
                    seeds: 4,
                    ..downstack::Spec::default()
                },
                bot: s("dt20"),
            },
        ),
    ]
}

/// Look a registered experiment up by name.
pub fn find(name: &str) -> Option<Entry> {
    entries().into_iter().find(|e| e.name == name)
}

/// The spec exactly as `show` prints it and the receipt records it.
pub fn spec_json(experiment: &Experiment) -> serde_json::Value {
    serde_json::to_value(experiment).expect("registry specs serialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots;

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

    /// Every bot a binding references must resolve — a typo here would
    /// otherwise die at run time instead of in CI.
    #[test]
    fn every_bound_bot_resolves() {
        use Experiment::*;
        let check = |entry: &str, name: &str| {
            assert!(
                bots::find(name).is_some(),
                "{entry}: bot {name:?} is not registered"
            );
        };
        for entry in entries() {
            match &entry.experiment {
                Marathon { bot, .. }
                | Downstack { bot, .. }
                | Behavior { bot, .. }
                | Awareness { bot, .. } => check(entry.name, bot),
                Versus { a, b, .. } => {
                    check(entry.name, a);
                    check(entry.name, b);
                }
                Race {
                    candidate,
                    incumbent,
                    ..
                } => {
                    check(entry.name, candidate);
                    check(entry.name, incumbent);
                }
                Panel { spec, candidate } => {
                    check(entry.name, candidate);
                    for opp in spec.must_beat.iter().chain(&spec.must_not_lose_to) {
                        check(entry.name, opp);
                    }
                }
                Climb { spec } => check(entry.name, &spec.subject),
                Cc2Baseline { .. } => {}
            }
        }
    }

    /// Bot names are unique too (the other registry).
    #[test]
    fn bot_names_are_unique() {
        let all = bots::bots();
        let mut names: Vec<_> = all.iter().map(|(n, _)| *n).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate bot names");
    }

    /// Every entry must serialize (it IS the receipt format) and round-trip
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
            "smoke-panel",
            "smoke-downstack",
            "downstack-metric",
            "app-metric",
            "cc2-board-climb",
            "panel-null-check",
        ] {
            assert!(find(name).is_some(), "missing load-bearing entry {name}");
        }
    }
}
