//! THE eval registry: every runnable measurement, by name, as code.
//!
//! An entry is a named, typed eval spec; bots are typed at the prompt
//! (`run <eval> [bots…]` — see [`crate::bots`] for their registry). Nothing
//! else configures an experiment — no environment variables, no config
//! files, no per-knob flags. There is no `list`/`show`: this FILE is the
//! catalog, read it.
//!
//! The discipline this buys: **a recorded result reproduces from
//! `(commit, eval, bots…)`** — all three are names, all three land in the
//! run receipt. Changing anything that could change a result means
//! registering a new name (one literal, versioned with the code), never
//! mutating a name that has recorded runs — `resume` refuses a checkpoint
//! whose stored spec no longer matches its entry, and a dirty working tree
//! is stamped into every receipt.
//!
//! Optimizers are not evals: the search side was removed pending a
//! first-principles redesign (history in git, `aa7bda9` and earlier).

use crate::commands::{cc2_baseline, downstack, marathon, race, versus};

/// One runnable eval: a name, a one-line description, and its spec.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: &'static str,
    pub about: &'static str,
    pub experiment: Experiment,
}

/// Every eval kind the platform runs, with its typed spec. Serializes
/// internally tagged — the form the receipt records.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Experiment {
    Marathon(marathon::Spec),
    Downstack(downstack::Spec),
    Versus(versus::Spec),
    Race(race::Spec),
    Cc2Baseline(cc2_baseline::Spec),
}

impl Experiment {
    /// How many bots `run` must be given for this eval.
    pub fn bot_slots(&self) -> usize {
        match self {
            Experiment::Versus(_) | Experiment::Race(_) => 2,
            Experiment::Cc2Baseline(_) => 0,
            _ => 1,
        }
    }

    /// The positional-argument shape, for error messages.
    pub fn usage(&self) -> &'static str {
        match self {
            Experiment::Versus(_) => "<bot-a> <bot-b>",
            Experiment::Race(_) => "<candidate> <incumbent>",
            Experiment::Cc2Baseline(_) => "",
            _ => "<bot>",
        }
    }
}

fn e(name: &'static str, about: &'static str, experiment: Experiment) -> Entry {
    Entry {
        name,
        about,
        experiment,
    }
}

/// The catalog. Names with recorded runs are permanent.
pub fn entries() -> Vec<Entry> {
    use Experiment::*;
    vec![
        // --- evals (bots at the prompt) -------------------------------------
        e(
            "marathon",
            "capped-marathon score/sec + APP (autoresearch: `run marathon dt20`)",
            Marathon(marathon::Spec::default()),
        ),
        e(
            "downstack",
            "censored cheese pieces + clear rate (autoresearch: `run downstack dt20`)",
            Downstack(downstack::Spec::default()),
        ),
        e(
            "versus",
            "head-to-head win/death/attack report (`run versus cc2-default dt20`)",
            Versus(versus::Spec::default()),
        ),
        e(
            "race",
            "pair-GSPRT survival verdict (`run race <candidate> attack-tuned`)",
            Race(race::Spec::default()),
        ),
        // --- external referee ------------------------------------------------
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
        // --- smoke (the research-smoke gate; deliberately tiny) --------------
        e(
            "smoke-downstack",
            "downstack at 4 seeds (the parse-contract canary)",
            Downstack(downstack::Spec {
                seeds: 4,
                ..downstack::Spec::default()
            }),
        ),
    ]
}

/// Look a registered eval up by name.
pub fn find(name: &str) -> Option<Entry> {
    entries().into_iter().find(|e| e.name == name)
}

/// The spec exactly as the receipt records it.
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
    fn every_entry_serializes_and_resolves() {
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

    /// The smoke gate, autoresearch loops, and docs reference these by name;
    /// renaming them breaks scripts silently — fail here instead.
    #[test]
    fn load_bearing_names_exist() {
        for name in ["smoke-downstack", "marathon", "downstack", "versus", "race"] {
            assert!(find(name).is_some(), "missing load-bearing entry {name}");
        }
    }
}
