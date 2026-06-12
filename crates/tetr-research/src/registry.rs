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
//! Two deliberate exceptions to bots-at-the-prompt: the climb names its
//! `subject` in-spec (a campaign's origin is pinned — the same campaign must
//! never climb from two origins), and the panel names its opponent bars
//! in-spec (they define the gate; only the candidate is yours to pass).

use crate::commands::{
    awareness, behavior, cc2_baseline, climb, downstack, marathon, panel, race, versus,
};

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
    Behavior(behavior::Spec),
    Awareness(awareness::Spec),
    Race(race::Spec),
    Panel(panel::Spec),
    Climb(climb::Spec),
    Cc2Baseline(cc2_baseline::Spec),
}

impl Experiment {
    /// How many bots `run` must be given for this eval.
    pub fn bot_slots(&self) -> usize {
        match self {
            Experiment::Versus(_) | Experiment::Race(_) => 2,
            Experiment::Climb(_) | Experiment::Cc2Baseline(_) => 0,
            _ => 1,
        }
    }

    /// The positional-argument shape, for error messages.
    pub fn usage(&self) -> &'static str {
        match self {
            Experiment::Versus(_) => "<bot-a> <bot-b>",
            Experiment::Race(_) => "<candidate> <incumbent>",
            Experiment::Climb(_) | Experiment::Cc2Baseline(_) => "",
            Experiment::Panel(_) => "<candidate>",
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

fn s(name: &str) -> String {
    name.to_string()
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
            "behavior",
            "APP / DS-P suite across the standard garbage scenarios",
            Behavior(behavior::Spec::default()),
        ),
        e(
            "awareness",
            "garbage-awareness A/B: the bot vs its blinded twin, arm-swapped",
            Awareness(awareness::Spec::default()),
        ),
        e(
            "race",
            "pair-GSPRT survival verdict (`run race <candidate> attack-tuned`)",
            Race(race::Spec::default()),
        ),
        e(
            "panel",
            "promotion panel: candidate vs in-spec opponent bars (scratch campaign)",
            Panel(panel::Spec::default()),
        ),
        // --- optimizers (subject pinned in-spec) ----------------------------
        e(
            "cc2-board-climb",
            "the production (1+1)-ES board-param climb from attack-tuned: \
             v3-calibrated screen (margin 150, sigma 0.08), confirm α=0.02, \
             anchored — campaign cc2-board-v4",
            Climb(climb::Spec {
                campaign: s("cc2-board-v4"),
                accept_margin: 150.0,
                sigma: 0.08,
                ..climb::Spec::default()
            }),
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
            "smoke-climb",
            "tiny anchored climb for the smoke gate (seconds; campaign smoke)",
            Climb(climb::Spec {
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
            }),
        ),
        e(
            "smoke-panel",
            "tiny promotion panel for the smoke gate (starved cells must REJECT)",
            Panel(panel::Spec {
                campaign: s("smoke"),
                must_beat: vec![s("greedy")],
                must_not_lose_to: vec![s("attack-tuned-tiny")],
                cell_matches: 4,
                max_plies: 16,
                ..panel::Spec::default()
            }),
        ),
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

    /// The two in-spec bot references (climb subjects, panel bars) must
    /// resolve — a typo here would otherwise die mid-run instead of in CI.
    #[test]
    fn in_spec_bot_names_resolve() {
        let check = |entry: &str, name: &str| {
            assert!(
                bots::find(name).is_some(),
                "{entry}: bot {name:?} is not registered"
            );
        };
        for entry in entries() {
            match &entry.experiment {
                Experiment::Climb(spec) => check(entry.name, &spec.subject),
                Experiment::Panel(spec) => {
                    for opp in spec.must_beat.iter().chain(&spec.must_not_lose_to) {
                        check(entry.name, opp);
                    }
                }
                _ => {}
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
        for name in [
            "smoke-climb",
            "smoke-panel",
            "smoke-downstack",
            "marathon",
            "downstack",
            "race",
            "panel",
            "cc2-board-climb",
        ] {
            assert!(find(name).is_some(), "missing load-bearing entry {name}");
        }
    }
}
