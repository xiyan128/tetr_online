//! Level-goal progression and the Back-to-Back qualification rules.
//!
//! Two goal systems are supported: Fixed (prorated start level, then ten lines
//! per level) and Variable (per-clear "goal units" weighted by clear type, §25.9).
//! [`GoalProgress`] tracks the current level and lines remaining and advances on
//! [`GoalProgress::award`]. The free functions also expose which clears qualify
//! for or break a Back-to-Back chain, shared with [`scoring`](crate::engine::scoring).

use crate::engine::gravity::{MAX_LEVEL, MIN_LEVEL};
use crate::engine::t_spin::TSpinKind;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GoalSystem {
    Fixed,
    Variable,
    /// No goal and no level progression: the level (and so gravity and the
    /// score multiplier) stays at the starting level forever. The versus
    /// convention — pressure comes from the opponent, not the clock. The goal
    /// is permanently `0` lines remaining, which [`GoalProgress::award`]
    /// already treats as "nothing to advance".
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoalProgress {
    system: GoalSystem,
    start_level: u8,
    level: u8,
    remaining: usize,
}

impl GoalProgress {
    pub fn new(system: GoalSystem, start_level: u8) -> Self {
        let start_level = clamp_level(start_level);
        Self {
            system,
            start_level,
            level: start_level,
            remaining: goal_for_level(system, start_level, start_level),
        }
    }

    pub fn level(&self) -> u8 {
        self.level
    }

    /// Test-only: rewind to the starting level and its goal, discarding any
    /// progress. Lets the acceptance suite reproduce the §13.3 example's explicit
    /// "At Level 1" precondition across a chain longer than one level's goal
    /// (§14.2 would otherwise level up after 10 line clears). No production code
    /// path calls this; it adds no goal/level behavior of its own.
    #[doc(hidden)]
    pub fn reset_to_start(&mut self) {
        self.level = self.start_level;
        self.remaining = goal_for_level(self.system, self.start_level, self.start_level);
    }

    pub fn remaining(&self) -> usize {
        self.remaining
    }

    pub fn award(&mut self, units: usize) -> u8 {
        if self.remaining == 0 {
            return 0;
        }

        let mut units = units;
        let mut levels_advanced = 0;

        while units >= self.remaining && self.remaining > 0 {
            if self.level >= MAX_LEVEL {
                self.remaining = 0;
                return levels_advanced;
            }

            units -= self.remaining;
            self.level += 1;
            levels_advanced += 1;
            self.remaining = goal_for_level(self.system, self.start_level, self.level);
        }

        if self.remaining > 0 {
            self.remaining -= units;
        }
        levels_advanced
    }
}

pub fn goal_for_level(system: GoalSystem, start_level: u8, level: u8) -> usize {
    let start_level = clamp_level(start_level);
    let level = clamp_level(level);
    match system {
        GoalSystem::Fixed => fixed_goal_for_level(start_level, level),
        GoalSystem::Variable => variable_goal_for_level(level),
        GoalSystem::None => 0,
    }
}

pub fn fixed_goal_for_level(start_level: u8, level: u8) -> usize {
    let start_level = clamp_level(start_level);
    let level = clamp_level(level);
    if level == start_level {
        10 * start_level as usize
    } else {
        10
    }
}

pub fn variable_goal_for_level(level: u8) -> usize {
    clamp_level(level) as usize * 5
}

pub fn variable_goal_units(t_spin: Option<TSpinKind>, lines: usize, back_to_back: bool) -> usize {
    let base_units = match (t_spin, lines) {
        (None, 0) => 0,
        (None, 1) => 1,
        (None, 2) => 3,
        (None, 3) => 5,
        (None, 4) => 8,
        (Some(TSpinKind::Mini), 0) => 1,
        (Some(TSpinKind::Mini), 1) => 2,
        (Some(TSpinKind::Mini), 2) => 4, // Mini Double: score 400 => 4 units (the score/100 pattern)
        (Some(TSpinKind::Full), 0) => 4,
        (Some(TSpinKind::Full), 1) => 8,
        (Some(TSpinKind::Full), 2) => 12,
        (Some(TSpinKind::Full), 3) => 16,
        _ => 0,
    };

    if back_to_back && qualifies_for_back_to_back(t_spin, lines) {
        base_units + base_units / 2
    } else {
        base_units
    }
}

pub fn qualifies_for_back_to_back(t_spin: Option<TSpinKind>, lines: usize) -> bool {
    // Back-to-Back: a Tetris or ANY T-spin line clear (Mini Single/Double included,
    // per the guideline's "difficult clears" rule).
    matches!(
        (t_spin, lines),
        (None, 4) | (Some(TSpinKind::Mini), 1..=2) | (Some(TSpinKind::Full), 1..=3)
    )
}

pub fn breaks_back_to_back(t_spin: Option<TSpinKind>, lines: usize) -> bool {
    t_spin.is_none() && matches!(lines, 1..=3)
}

fn clamp_level(level: u8) -> u8 {
    level.clamp(MIN_LEVEL, MAX_LEVEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_goal_matches_section_25_9_examples() {
        assert_eq!(fixed_goal_for_level(1, 1), 10);
        assert_eq!(fixed_goal_for_level(4, 4), 40);
        assert_eq!(fixed_goal_for_level(4, 5), 10);
    }

    #[test]
    fn variable_goal_matches_section_25_9_examples() {
        assert_eq!(variable_goal_for_level(1), 5);
        assert_eq!(variable_goal_for_level(15), 75);
        assert_eq!((1..=15).map(variable_goal_for_level).sum::<usize>(), 600);
    }

    #[test]
    fn variable_goal_units_match_guideline_table() {
        assert_eq!(variable_goal_units(None, 1, false), 1);
        assert_eq!(variable_goal_units(None, 2, false), 3);
        assert_eq!(variable_goal_units(None, 3, false), 5);
        assert_eq!(variable_goal_units(None, 4, false), 8);
        assert_eq!(variable_goal_units(Some(TSpinKind::Mini), 0, false), 1);
        assert_eq!(variable_goal_units(Some(TSpinKind::Mini), 1, false), 2);
        assert_eq!(variable_goal_units(Some(TSpinKind::Mini), 2, false), 4);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 0, false), 4);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 1, false), 8);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 2, false), 12);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 3, false), 16);
    }

    #[test]
    fn variable_goal_units_apply_b2b_bonus_only_to_qualifying_actions() {
        assert_eq!(variable_goal_units(None, 4, true), 12);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 2, true), 18);
        assert_eq!(variable_goal_units(Some(TSpinKind::Full), 0, true), 4);
        assert_eq!(variable_goal_units(None, 2, true), 3);
        assert_eq!(variable_goal_units(Some(TSpinKind::Mini), 2, true), 6); // 4 + 4/2
    }

    #[test]
    fn every_t_spin_line_clear_qualifies_for_back_to_back() {
        // The guideline "difficult clears" rule: Tetris + every T-spin LINE clear
        // (Mini Double included — the row the tables used to disagree on).
        assert!(qualifies_for_back_to_back(None, 4));
        assert!(qualifies_for_back_to_back(Some(TSpinKind::Mini), 1));
        assert!(qualifies_for_back_to_back(Some(TSpinKind::Mini), 2));
        assert!(qualifies_for_back_to_back(Some(TSpinKind::Full), 1));
        assert!(qualifies_for_back_to_back(Some(TSpinKind::Full), 3));
        // Zero-line spins and plain 1-3 line clears do not qualify.
        assert!(!qualifies_for_back_to_back(Some(TSpinKind::Mini), 0));
        assert!(!qualifies_for_back_to_back(Some(TSpinKind::Full), 0));
        assert!(!qualifies_for_back_to_back(None, 1));
        assert!(!qualifies_for_back_to_back(None, 3));
        // And plain small clears BREAK the chain while spins/no-clears preserve it.
        assert!(breaks_back_to_back(None, 2));
        assert!(!breaks_back_to_back(Some(TSpinKind::Mini), 2));
        assert!(!breaks_back_to_back(None, 0));
    }

    #[test]
    fn fixed_goal_progress_prorates_start_level_then_uses_ten_lines() {
        let mut progress = GoalProgress::new(GoalSystem::Fixed, 4);

        assert_eq!(progress.remaining(), 40);
        assert_eq!(progress.award(39), 0);
        assert_eq!(progress.level(), 4);
        assert_eq!(progress.remaining(), 1);

        assert_eq!(progress.award(2), 1);
        assert_eq!(progress.level(), 5);
        assert_eq!(progress.remaining(), 9);
    }

    #[test]
    fn none_goal_system_never_levels() {
        let mut progress = GoalProgress::new(GoalSystem::None, 1);

        assert_eq!(progress.remaining(), 0, "no goal exists to chase");
        // A marathon's worth of clears advances nothing.
        assert_eq!(progress.award(200), 0);
        assert_eq!(progress.level(), 1, "the level is pinned at the start");
        assert_eq!(progress.remaining(), 0);
    }

    #[test]
    fn variable_goal_progress_uses_next_level_goal_after_level_up() {
        let mut progress = GoalProgress::new(GoalSystem::Variable, 1);

        assert_eq!(progress.remaining(), 5);
        assert_eq!(progress.award(5), 1);
        assert_eq!(progress.level(), 2);
        assert_eq!(progress.remaining(), 10);
    }

    #[test]
    fn goal_progress_does_not_underflow_after_max_level_completion() {
        let mut progress = GoalProgress::new(GoalSystem::Variable, MAX_LEVEL);

        assert_eq!(progress.award(75), 0);
        assert_eq!(progress.level(), MAX_LEVEL);
        assert_eq!(progress.remaining(), 0);
        assert_eq!(progress.award(1), 0);
        assert_eq!(progress.remaining(), 0);
    }
}
