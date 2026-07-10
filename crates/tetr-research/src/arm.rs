//! An **arm**: one player in a versus instrument, named by a small grammar
//! and parsed exactly once.
//!
//! ```text
//! greedy                    the shipped greedy baseline
//! beam:cc2@w8d5             beam search, CC2 hand eval, width 8 depth 5
//! tp:cc2@w128d9             transposition-pruned beam, CC2 hand eval
//! beam:<model-dir>@w8d5     beam search with the net as the leaf evaluator
//! ```
//!
//! An arm string is an experiment's *identity* (it goes in the receipt
//! verbatim), and every arm builds through the same full-strength convention
//! as the bot registry — so any two arms are apples-to-apples by construction.
//!
//! The grammar once carried five more net-arm forms (policy/value probes and
//! two filtered "guided" vehicles); they measured heads the current net does
//! not have and their hidden dispatch voided a whole campaign. History in
//! `wayfinder/leapfrog/archive/`.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use tetr_core::ai::BeamPlanner;
use tetr_core::ai::SearchBudget;
use tetr_core::ai::eval::Cc2Weights;
use tetr_core::player::PlayerController;
use tetr_nn::serve::NetEvaluator;

use crate::bots::{BotSpec, full_strength};

/// A parsed arm. `Clone` so instruments can fan factories across threads.
/// (No `PartialEq`: `BotSpec` deliberately compares by recorded identity in
/// its own ways; arms compare by their display string when needed.)
#[derive(Clone, Debug)]
pub enum Arm {
    /// A hand-eval bot from the registry's spec vocabulary.
    Spec(BotSpec),
    /// Beam search with the net as the leaf evaluator (pure `z_scale · z_hat`
    /// scores — no hand terms).
    NetBeam {
        dir: PathBuf,
        width: usize,
        depth: u8,
    },
}

impl Arm {
    /// Build a fresh controller (policy RNG seeded by `seed`).
    pub fn controller(&self, seed: u64) -> Box<dyn PlayerController> {
        match self {
            Arm::Spec(spec) => spec.controller(seed),
            Arm::NetBeam { dir, width, depth } => full_strength(
                Box::new(BeamPlanner::new(*width)),
                Box::new(NetEvaluator::load(dir).expect("arm model dir loads")),
                SearchBudget::beam(*depth),
                seed,
            ),
        }
    }

    /// This arm as a harness factory — what the one versus loop takes.
    pub fn factory(&self) -> impl Fn(u64) -> Box<dyn PlayerController> + Send + Sync + use<> {
        let arm = self.clone();
        move |seed| arm.controller(seed)
    }
}

impl fmt::Display for Arm {
    /// The receipt identity: round-trips the grammar (a parsed arm displays
    /// as a string the parser accepts).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::bots::{EvalSpec, SearchSpec};
        // The eval tag half of a hand-eval arm string.
        let eval_tag = |eval| {
            if matches!(eval, EvalSpec::Cc2(_)) {
                "cc2"
            } else {
                "linear"
            }
        };
        match self {
            Arm::Spec(spec) => match (spec.search, spec.eval) {
                (SearchSpec::Greedy, _) => write!(f, "greedy"),
                (SearchSpec::Beam { width, depth }, eval) => {
                    write!(f, "beam:{}@w{width}d{depth}", eval_tag(eval))
                }
                (SearchSpec::TpBeam { width, depth }, eval) => {
                    write!(f, "tp:{}@w{width}d{depth}", eval_tag(eval))
                }
                _ => write!(f, "{spec:?}"),
            },
            Arm::NetBeam { dir, width, depth } => {
                write!(f, "beam:{}@w{width}d{depth}", dir.display())
            }
        }
    }
}

/// Parse a `w<width>d<depth>` suffix.
fn parse_wd(s: &str) -> Result<(usize, u8), String> {
    let rest = s
        .strip_prefix('w')
        .ok_or_else(|| format!("expected w<width>d<depth>, got {s:?}"))?;
    let (w, d) = rest
        .split_once('d')
        .ok_or_else(|| format!("expected w<width>d<depth>, got {s:?}"))?;
    Ok((
        w.parse().map_err(|_| format!("bad width in {s:?}"))?,
        d.parse().map_err(|_| format!("bad depth in {s:?}"))?,
    ))
}

impl FromStr for Arm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if s == "greedy" {
            return Ok(Arm::Spec(BotSpec::greedy()));
        }
        let (kind, rest) = s
            .split_once(':')
            .ok_or_else(|| format!("arm {s:?}: expected kind:target (or `greedy`)"))?;
        match kind {
            "beam" | "tp" => {
                let (target, cfg) = rest
                    .rsplit_once('@')
                    .ok_or_else(|| format!("arm {s:?}: expected {kind}:target@w<W>d<D>"))?;
                let (width, depth) = parse_wd(cfg)?;
                match target {
                    // ONE CC2 everywhere: the attack-tuned (champion-family)
                    // weights — the same evaluator datagen's round-0 teacher
                    // uses, so "cc2" never means two different bots.
                    "cc2" => {
                        let spec = if kind == "tp" {
                            BotSpec::tp_beam(width, depth)
                        } else {
                            BotSpec::beam(width, depth)
                        };
                        Ok(Arm::Spec(spec.cc2(Cc2Weights::attack_tuned())))
                    }
                    "linear" => Ok(Arm::Spec(if kind == "tp" {
                        BotSpec::tp_beam(width, depth)
                    } else {
                        BotSpec::beam(width, depth)
                    })),
                    dir if kind == "beam" => Ok(Arm::NetBeam {
                        dir: dir.into(),
                        width,
                        depth,
                    }),
                    _ => Err(format!(
                        "arm {s:?}: a net beam has no TP variant (yet) — use beam:<dir>@…"
                    )),
                }
            }
            other => Err(format!(
                "arm {s:?}: unknown kind {other:?} (greedy | beam | tp)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grammar_parses_every_documented_form() {
        assert!(matches!("greedy".parse::<Arm>(), Ok(Arm::Spec(_))));
        assert!(matches!("beam:cc2@w8d5".parse::<Arm>(), Ok(Arm::Spec(_))));
        assert!(matches!("tp:cc2@w128d9".parse::<Arm>(), Ok(Arm::Spec(_))));
        match "beam:models/round0@w8d5".parse::<Arm>() {
            Ok(Arm::NetBeam { dir, width, depth }) => {
                assert_eq!(
                    (dir.to_str().unwrap(), width, depth),
                    ("models/round0", 8, 5)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn junk_is_rejected_with_a_named_reason() {
        for junk in [
            "",
            "beam",
            "beam:cc2",
            "beam:cc2@8x5",
            "tp:models/x@w8d5",
            "warp:x",
            "policy:models/x",
            "value:models/x",
            "guided:models/x@m12w8d5",
        ] {
            assert!(junk.parse::<Arm>().is_err(), "{junk:?} must not parse");
        }
    }

    proptest::proptest! {
        /// The parser is total: no input string panics — it parses or errors.
        #[test]
        fn parsing_is_total(s in ".*") {
            let _ = s.parse::<Arm>();
        }

        /// A hand-eval arm's receipt identity is stable under a round-trip:
        /// parse -> Display -> parse yields the same Display.
        #[test]
        fn hand_eval_arms_round_trip(
            kind in proptest::sample::select(vec!["beam", "tp"]),
            eval in proptest::sample::select(vec!["cc2", "linear"]),
            w in 1usize..512,
            d in 1u8..20,
        ) {
            let s = format!("{kind}:{eval}@w{w}d{d}");
            let disp = s.parse::<Arm>().unwrap().to_string();
            let reparsed = disp.parse::<Arm>().unwrap().to_string();
            proptest::prop_assert_eq!(disp, reparsed);
        }

        /// A net arm displays back to exactly the string it parsed from (the
        /// dir is constrained to a path so it can't collide with cc2/linear).
        #[test]
        fn net_arms_round_trip_exactly(
            dir in "models/[a-z0-9]{1,10}",
            w in 1usize..256,
            d in 1u8..16,
        ) {
            let s = format!("beam:{dir}@w{w}d{d}");
            let arm = s.parse::<Arm>().unwrap();
            proptest::prop_assert_eq!(&arm.to_string(), &s);
        }
    }
}
