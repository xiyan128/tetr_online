//! The search side of the platform: optimizers and their promotion gate,
//! behind the `tetr-climb` binary — NOT evals. A climb MUTATES a subject
//! bot's weights against an objective; a panel decides whether the result
//! deserves to be called better. Both wrap the measurement primitives
//! ([`crate::sprt`], the versus harness) and the same receipt/campaign
//! discipline as everything else.
//!
//! Configurations are named literals, same rules as the eval registry:
//! result-affecting change = new name; names with recorded runs are
//! permanent. This file is the catalog.

use crate::versus::VersusFormat;

pub mod climb;
pub mod panel;

/// Named climb configurations (`tetr-climb climb <name>`).
pub fn climbs() -> Vec<(&'static str, climb::Spec)> {
    vec![
        (
            // The production board-param climb: v3-calibrated screen
            // (margin 150, sigma 0.08), confirm α=0.02, anchored.
            "cc2-board-v4",
            climb::Spec {
                campaign: "cc2-board-v4".to_string(),
                accept_margin: 150.0,
                sigma: 0.08,
                ..climb::Spec::default()
            },
        ),
        (
            // The smoke gate's seconds-sized walk.
            "smoke",
            climb::Spec {
                campaign: "smoke".to_string(),
                subject: "attack-tuned-tiny".to_string(),
                format: VersusFormat {
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
        ),
    ]
}

/// Named panel configurations (`tetr-climb panel <name> <candidate>`).
pub fn panels() -> Vec<(&'static str, panel::Spec)> {
    vec![
        ("default", panel::Spec::default()),
        (
            // The smoke gate's toy panel: starved cells must REJECT.
            "smoke",
            panel::Spec {
                campaign: "smoke".to_string(),
                must_beat: vec!["greedy".to_string()],
                must_not_lose_to: vec!["attack-tuned-tiny".to_string()],
                cell_matches: 4,
                max_plies: 16,
                ..panel::Spec::default()
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots;

    /// Every in-spec bot reference (climb subjects, panel bars) must resolve.
    #[test]
    fn in_spec_bot_names_resolve() {
        for (name, spec) in climbs() {
            assert!(
                bots::find(&spec.subject).is_some(),
                "climb {name}: subject {:?} is not a registered bot",
                spec.subject
            );
        }
        for (name, spec) in panels() {
            for opp in spec.must_beat.iter().chain(&spec.must_not_lose_to) {
                assert!(
                    bots::find(opp).is_some(),
                    "panel {name}: opponent {opp:?} is not a registered bot"
                );
            }
        }
    }

    /// The smoke gate runs these by name.
    #[test]
    fn load_bearing_names_exist() {
        assert!(climbs().iter().any(|(n, _)| *n == "smoke"));
        assert!(climbs().iter().any(|(n, _)| *n == "cc2-board-v4"));
        assert!(panels().iter().any(|(n, _)| *n == "smoke"));
        assert!(panels().iter().any(|(n, _)| *n == "default"));
    }
}
