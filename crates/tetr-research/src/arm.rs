//! An **arm**: one player in a versus instrument, named by a small grammar
//! and parsed exactly once.
//!
//! ```text
//! greedy                    the shipped greedy baseline
//! beam:cc2@w8d5             beam search, CC2 hand eval, width 8 depth 5
//! tp:cc2@w128d9             transposition-pruned beam, CC2 hand eval
//! beam:<model-dir>@w8d5     beam search with the net as the leaf evaluator
//! value:<model-dir>         depth-1 value argmax over the net (full obs)
//! policy:<model-dir>        the net's policy head, argmax at the root
//! ```
//!
//! An arm string is an experiment's *identity* (it goes in the receipt
//! verbatim), and every arm builds through the same full-strength convention
//! as the bot registry — so any two arms are apples-to-apples by
//! construction. This one grammar replaces the reference campaign's ArmB
//! enums and its repurposed `--infer` flag.
//!
//! Measurement idioms: `beam:M@w8d5` vs `policy:M` is the G_π probe (how much
//! does search improve the net's own policy); `value:M` is the d1-value arm
//! of the same family; candidate-beam vs incumbent-beam is a strength race.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use tetr_core::ai::eval::{Cc2Weights, Evaluator, LinearEvaluator};
use tetr_core::ai::movegen::Placement;
use tetr_core::ai::search::{Mind, PlacementPlan, ThinkProgress, hold_placements};
use tetr_core::ai::state::SearchState;
use tetr_core::ai::{BeamPlanner, SearchBudget};
use tetr_core::player::PlayerController;
use tetr_nn::net::{BoardEmb, Net, Scratch};
use tetr_nn::obs::{Obs, OppCtx, encode};
use tetr_nn::serve::NetEvaluator;

use crate::bots::{BotSpec, full_strength};

/// A parsed arm. `Clone` so instruments can fan factories across threads.
/// (No `PartialEq`: `BotSpec` deliberately compares by recorded identity in
/// its own ways; arms compare by their display string when needed.)
#[derive(Clone, Debug)]
pub enum Arm {
    /// A hand-eval bot from the registry's spec vocabulary.
    Spec(BotSpec),
    /// Beam search with the net as the leaf evaluator.
    NetBeam {
        dir: PathBuf,
        width: usize,
        depth: u8,
    },
    /// Depth-1 value argmax over the net, full observations (a width-1 beam at
    /// depth 1 — the seeding scores every root, so the decision is the exact
    /// argmax regardless of width).
    NetValue { dir: PathBuf },
    /// The net's policy head, argmax over the root's children. No search.
    NetPolicy { dir: PathBuf },
    /// The policy-GUIDED beam (the deployed-vehicle seed): the net's policy
    /// head picks the top-m roots, the beam searches only those with the net
    /// as leaf evaluator. TP on (width buys distinct futures).
    GuidedBeam {
        dir: PathBuf,
        m: usize,
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
            Arm::NetValue { dir } => full_strength(
                Box::new(BeamPlanner::new(1)),
                Box::new(NetEvaluator::load(dir).expect("arm model dir loads")),
                SearchBudget::beam(1),
                seed,
            ),
            Arm::NetPolicy { dir } => full_strength(
                Box::new(PolicyMind::load(dir)),
                // The mind carries its own net; this eval slot is inert (the
                // session runner requires one).
                Box::new(LinearEvaluator::default()),
                SearchBudget::beam(1),
                seed,
            ),
            Arm::GuidedBeam {
                dir,
                m,
                width,
                depth,
            } => full_strength(
                Box::new(BeamPlanner::transposing(*width).with_root_filter(policy_top_m(dir, *m))),
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
            Arm::NetValue { dir } => write!(f, "value:{}", dir.display()),
            Arm::NetPolicy { dir } => write!(f, "policy:{}", dir.display()),
            Arm::GuidedBeam {
                dir,
                m,
                width,
                depth,
            } => write!(f, "guided:{}@m{m}w{width}d{depth}", dir.display()),
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
                    "cc2" => {
                        let spec = if kind == "tp" {
                            BotSpec::tp_beam(width, depth)
                        } else {
                            BotSpec::beam(width, depth)
                        };
                        Ok(Arm::Spec(spec.cc2(Cc2Weights::default())))
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
            "value" => Ok(Arm::NetValue { dir: rest.into() }),
            "policy" => Ok(Arm::NetPolicy { dir: rest.into() }),
            "guided" => {
                let (target, cfg) = rest
                    .rsplit_once('@')
                    .ok_or_else(|| format!("arm {s:?}: expected guided:dir@m<M>w<W>d<D>"))?;
                let cfg = cfg
                    .strip_prefix('m')
                    .ok_or_else(|| format!("arm {s:?}: expected m<M>w<W>d<D>"))?;
                let (m_str, wd) = cfg
                    .split_once('w')
                    .ok_or_else(|| format!("arm {s:?}: expected m<M>w<W>d<D>"))?;
                let (width, depth) = parse_wd(&format!("w{wd}"))?;
                Ok(Arm::GuidedBeam {
                    dir: target.into(),
                    m: m_str.parse().map_err(|_| format!("bad m in {s:?}"))?,
                    width,
                    depth,
                })
            }
            other => Err(format!(
                "arm {s:?}: unknown kind {other:?} (greedy | beam | tp | value | policy | guided)"
            )),
        }
    }
}

/// A [`Mind`] that plays the net's policy head: at each root, encode every
/// child (under a neutral opponent — the policy arm is deliberately blind, it
/// measures the *prior*, not opponent modeling), one batched forward, argmax
/// policy logit. Dead children are excluded; ties keep the first (canonical
/// movegen order), matching every other planner's tie rule.
struct PolicyMind {
    net: Arc<Net>,
    opp: OppCtx,
    opp_emb: BoardEmb,
    scratch: Scratch,
    plan: Option<PlacementPlan>,
    expanded: u32,
}

impl PolicyMind {
    fn load(dir: &Path) -> Self {
        let net = Net::load(dir).expect("arm model dir loads");
        let mut scratch = Scratch::default();
        let opp = OppCtx::default();
        let opp_emb = net
            .embed_boards(&[&opp.board], &mut scratch)
            .pop()
            .expect("one plane in, one embedding out");
        Self {
            net: Arc::new(net),
            opp,
            opp_emb,
            scratch,
            plan: None,
            expanded: 0,
        }
    }
}

/// Build the policy-top-m [`RootFilter`]: one batched policy forward over the
/// state's children, keep the m highest-logit LIVE placements (dead children
/// never earn a beam root). Returns empty when every child is dead — the
/// planner's defensive fallback then searches all roots (never a manufactured
/// resignation). Deterministic per state; ties resolve to canonical movegen
/// order (stable sort), matching every other planner's tie rule.
pub fn policy_top_m(dir: &Path, m: usize) -> tetr_core::ai::search::RootFilter {
    let net = Net::load(dir).expect("guided-beam model dir loads");
    let mut scratch = Scratch::default();
    let opp = OppCtx::default();
    let opp_emb = net
        .embed_boards(&[&opp.board], &mut scratch)
        .pop()
        .expect("one plane in, one embedding out");
    let shared = std::sync::Mutex::new((net, scratch));
    Box::new(move |state: &SearchState, placements: Vec<Placement>| {
        let children: Vec<(Obs, bool)> = placements
            .iter()
            .map(|p| {
                let mut child = state.clone();
                child.commit_placement(p);
                (encode(&child, &opp), child.dead)
            })
            .collect();
        let items: Vec<_> = children
            .iter()
            .map(|(o, _)| (&o.own_board, &o.features))
            .collect();
        let mut guard = shared.lock().expect("filter net lock");
        let (net, scratch) = &mut *guard;
        let heads = net.forward(&items, &opp_emb, scratch);
        let mut live: Vec<(usize, f32)> = heads
            .iter()
            .zip(&children)
            .enumerate()
            .filter(|(_, (_, (_, dead)))| !dead)
            .map(|(i, (h, _))| (i, h.policy))
            .collect();
        // Stable by descending logit: ties keep canonical (movegen) order.
        live.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<usize> = live.into_iter().take(m).map(|(i, _)| i).collect();
        keep.sort_unstable(); // restore canonical order among the kept
        keep.into_iter().map(|i| placements[i].clone()).collect()
    })
}

impl Mind for PolicyMind {
    fn reroot(&mut self, state: &SearchState, _eval: &dyn Evaluator, _max_depth: u8) {
        let placements = hold_placements(state);
        self.expanded = placements.len() as u32;
        if placements.is_empty() {
            self.plan = None;
            return;
        }
        let children: Vec<(Obs, bool)> = placements
            .iter()
            .map(|p| {
                let mut child = state.clone();
                child.commit_placement(p);
                (encode(&child, &self.opp), child.dead)
            })
            .collect();
        let items: Vec<_> = children
            .iter()
            .map(|(o, _)| (&o.own_board, &o.features))
            .collect();
        let heads = self.net.forward(&items, &self.opp_emb, &mut self.scratch);

        let mut best: Option<(usize, f32)> = None;
        for (i, (h, (_, dead))) in heads.iter().zip(&children).enumerate() {
            if *dead {
                continue;
            }
            if best.is_none_or(|(_, b)| h.policy > b) {
                best = Some((i, h.policy));
            }
        }
        // A policy logit scales into the planner's integer score domain; an
        // all-children-dead root has no live logit, so it scores as a forced
        // loss (keeping the canonical-first move) rather than relying on
        // f32::MIN overflowing through the ×1000 cast.
        let (i, score) = match best {
            Some((i, logit)) => (i, (logit * 1000.0) as i32),
            None => (0, i32::MIN),
        };
        self.plan = Some(PlacementPlan {
            placement: placements[i].clone(),
            score,
        });
    }

    fn think(&mut self, _quantum: u32, _eval: &dyn Evaluator) -> ThinkProgress {
        ThinkProgress::Exhausted
    }

    fn best(&self) -> Option<PlacementPlan> {
        self.plan.clone()
    }

    fn nodes_expanded(&self) -> u32 {
        self.expanded
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
        assert!(matches!(
            "value:models/round0".parse::<Arm>(),
            Ok(Arm::NetValue { .. })
        ));
        assert!(matches!(
            "policy:models/round0".parse::<Arm>(),
            Ok(Arm::NetPolicy { .. })
        ));
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
            for s in [
                format!("beam:{dir}@w{w}d{d}"),
                format!("value:{dir}"),
                format!("policy:{dir}"),
            ] {
                let arm = s.parse::<Arm>().unwrap();
                proptest::prop_assert_eq!(&arm.to_string(), &s);
            }
        }
    }
}
