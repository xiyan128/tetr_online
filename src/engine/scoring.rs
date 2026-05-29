use crate::engine::goals::{
    breaks_back_to_back, qualifies_for_back_to_back, variable_goal_units, GoalProgress, GoalSystem,
};
use crate::engine::gravity::MIN_LEVEL;
use crate::engine::t_spin::TSpinKind;

/// Apply a scoring action to a `ScoreState` and return the resulting award, if
/// any.
///
/// Free function so future placement / replay code can run the scoring rules
/// against a borrowed state without having to construct an `Engine`. The
/// state-mutation surface is explicit (`&mut ScoreState`) per ADR-7.
///
/// `goal_system` is required only for lock-result actions; manual drops ignore
/// it.
pub(crate) fn score_action(
    state: &mut ScoreState,
    goal_system: GoalSystem,
    action: EngineScoreAction,
) -> Option<ScoreAward> {
    match action {
        EngineScoreAction::SoftDrop => state.manual_drop(action, 1),
        EngineScoreAction::HardDrop { cells } => state.manual_drop(action, cells),
        EngineScoreAction::NoClear => state.lock_result(goal_system, None, 0),
        EngineScoreAction::Single => state.lock_result(goal_system, None, 1),
        EngineScoreAction::Double => state.lock_result(goal_system, None, 2),
        EngineScoreAction::Triple => state.lock_result(goal_system, None, 3),
        EngineScoreAction::Tetris => state.lock_result(goal_system, None, 4),
        EngineScoreAction::TSpin { kind, lines } => {
            state.lock_result(goal_system, Some(kind), lines)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EngineScoreAction {
    SoftDrop,
    HardDrop { cells: usize },
    NoClear,
    Single,
    Double,
    Triple,
    Tetris,
    TSpin { kind: TSpinKind, lines: usize },
}

impl EngineScoreAction {
    pub(crate) fn from_lock_result(t_spin: Option<TSpinKind>, lines: usize) -> Self {
        if let Some(kind) = t_spin {
            return Self::TSpin { kind, lines };
        }

        match lines {
            0 => Self::NoClear,
            1 => Self::Single,
            2 => Self::Double,
            3 => Self::Triple,
            4 => Self::Tetris,
            _ => Self::NoClear,
        }
    }

    fn base_score(self, level: usize) -> usize {
        let base_score = match self {
            Self::SoftDrop | Self::HardDrop { .. } => 0,
            Self::NoClear => 0,
            Self::Single => 100,
            Self::Double => 300,
            Self::Triple => 500,
            Self::Tetris => 800,
            Self::TSpin {
                kind: TSpinKind::Mini,
                lines: 0,
            } => 100,
            Self::TSpin {
                kind: TSpinKind::Mini,
                lines: 1,
            } => 200,
            Self::TSpin {
                kind: TSpinKind::Full,
                lines: 0,
            } => 400,
            Self::TSpin {
                kind: TSpinKind::Full,
                lines: 1,
            } => 800,
            Self::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            } => 1200,
            Self::TSpin {
                kind: TSpinKind::Full,
                lines: 3,
            } => 1600,
            Self::TSpin { .. } => 0,
        };

        base_score * level
    }

    fn qualifies_for_back_to_back(self) -> bool {
        let (t_spin, lines) = self.spin_and_lines();
        qualifies_for_back_to_back(t_spin, lines)
    }

    fn breaks_back_to_back(self) -> bool {
        let (t_spin, lines) = self.spin_and_lines();
        breaks_back_to_back(t_spin, lines)
    }

    fn spin_and_lines(self) -> (Option<TSpinKind>, usize) {
        match self {
            Self::SoftDrop | Self::HardDrop { .. } => (None, 0),
            Self::NoClear => (None, 0),
            Self::Single => (None, 1),
            Self::Double => (None, 2),
            Self::Triple => (None, 3),
            Self::Tetris => (None, 4),
            Self::TSpin { kind, lines } => (Some(kind), lines),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScoreState {
    score: usize,
    lines: usize,
    back_to_back_active: bool,
    goal_progress: GoalProgress,
}

impl ScoreState {
    pub(crate) fn new(goal_system: GoalSystem, starting_level: u8) -> Self {
        Self {
            score: 0,
            lines: 0,
            back_to_back_active: false,
            goal_progress: GoalProgress::new(goal_system, starting_level),
        }
    }

    pub(crate) fn score(&self) -> usize {
        self.score
    }

    pub(crate) fn lines(&self) -> usize {
        self.lines
    }

    pub(crate) fn level(&self) -> u8 {
        self.goal_progress.level()
    }

    pub(crate) fn goal_remaining(&self) -> usize {
        self.goal_progress.remaining()
    }

    pub(crate) fn back_to_back_active(&self) -> bool {
        self.back_to_back_active
    }

    /// Test-only: rewind the goal/level progression to the starting level while
    /// preserving accumulated `score`, `lines`, and the Back-to-Back chain. Used
    /// by the acceptance suite to reproduce the §13.3 example's explicit
    /// "At Level 1" precondition across a chain longer than one level's goal.
    /// Adds no scoring behavior of its own.
    pub(crate) fn reset_level_for_test(&mut self) {
        self.goal_progress.reset_to_start();
    }

    pub(crate) fn lock_result(
        &mut self,
        goal_system: GoalSystem,
        t_spin: Option<TSpinKind>,
        lines_cleared: usize,
    ) -> Option<ScoreAward> {
        self.lines += lines_cleared;

        let action = EngineScoreAction::from_lock_result(t_spin, lines_cleared);
        let base_score = action.base_score(self.goal_progress.level() as usize);
        let back_to_back_bonus = action.qualifies_for_back_to_back() && self.back_to_back_active;
        let score = if back_to_back_bonus {
            base_score + base_score / 2
        } else {
            base_score
        };

        if action.qualifies_for_back_to_back() {
            self.back_to_back_active = true;
        } else if action.breaks_back_to_back() {
            self.back_to_back_active = false;
        }

        let goal_units = match goal_system {
            GoalSystem::Fixed => lines_cleared,
            GoalSystem::Variable => variable_goal_units(t_spin, lines_cleared, back_to_back_bonus),
        };
        self.goal_progress.award(goal_units);
        self.score += score;

        (score > 0).then_some(ScoreAward {
            action,
            score,
            total_score: self.score,
            back_to_back_bonus,
        })
    }

    pub(crate) fn manual_drop(
        &mut self,
        action: EngineScoreAction,
        cells: usize,
    ) -> Option<ScoreAward> {
        let score = match action {
            EngineScoreAction::SoftDrop => cells,
            EngineScoreAction::HardDrop { .. } => 2 * cells,
            _ => 0,
        };
        self.score += score;

        (score > 0).then_some(ScoreAward {
            action,
            score,
            total_score: self.score,
            back_to_back_bonus: false,
        })
    }
}

impl Default for ScoreState {
    fn default() -> Self {
        Self::new(GoalSystem::Fixed, MIN_LEVEL)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScoreAward {
    pub(crate) action: EngineScoreAction,
    pub(crate) score: usize,
    pub(crate) total_score: usize,
    pub(crate) back_to_back_bonus: bool,
}
