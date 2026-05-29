//! Shared player-settings contract (M1).
//!
//! [`GameSettings`] is the single source of truth for tunables the player can
//! change from the Options screen and that gameplay/render systems read:
//!
//! * `next_count` — how many previews to show (1..=6); feeds both
//!   [`EngineConfig::preview_count`](crate::engine::EngineConfig) (so the engine
//!   keeps the queue filled) and the on-screen previewer.
//! * `hold_enabled` / `ghost_enabled` — feature toggles read by gameplay/render.
//! * `lock_down_mode` — the engine [`LockDownMode`] used when building the engine.
//! * `music_volume` / `sfx_volume` — 0.0..=1.0, read by the SFX/music features.
//! * `keybinds` — the action→key map the keyboard controller reads.
//!
//! This type is defined ONCE here so the options feature mutates it and every
//! reader (engine bridge, previewer, ghost system, SFX) shares one definition.
//! Persistence is handled by the storage layer + options feature; this module
//! only owns the in-memory resource and its `Default`.

use bevy::prelude::*;

use crate::engine::LockDownMode;

/// The smallest and largest number of next-piece previews the UI/engine support.
pub const MIN_NEXT_COUNT: usize = 1;
pub const MAX_NEXT_COUNT: usize = 6;

/// A logical, rebindable player action. The keyboard controller maps the bound
/// [`KeyCode`]s to its raw input each frame; the Options screen rebinds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameAction {
    MoveLeft,
    MoveRight,
    SoftDrop,
    HardDrop,
    RotateCw,
    RotateCcw,
    Hold,
    Pause,
}

impl GameAction {
    /// All actions, in display order (used by the Options rebind list).
    pub const ALL: [GameAction; 8] = [
        GameAction::MoveLeft,
        GameAction::MoveRight,
        GameAction::SoftDrop,
        GameAction::HardDrop,
        GameAction::RotateCw,
        GameAction::RotateCcw,
        GameAction::Hold,
        GameAction::Pause,
    ];

    /// Human-readable label for the Options rebind list.
    pub fn label(self) -> &'static str {
        match self {
            GameAction::MoveLeft => "Move Left",
            GameAction::MoveRight => "Move Right",
            GameAction::SoftDrop => "Soft Drop",
            GameAction::HardDrop => "Hard Drop",
            GameAction::RotateCw => "Rotate CW",
            GameAction::RotateCcw => "Rotate CCW",
            GameAction::Hold => "Hold",
            GameAction::Pause => "Pause",
        }
    }
}

/// Action→keys binding map.
///
/// Each action may bind to a primary and an optional secondary [`KeyCode`]
/// (e.g. rotate-CW defaults to both `ArrowUp` and `KeyX`, matching the existing
/// hard-coded controller). The keyboard controller reads `pressed`/`just_pressed`
/// for *either* key.
#[derive(Resource, Debug, Clone, PartialEq, Eq)]
pub struct Keybinds {
    pub move_left: (KeyCode, Option<KeyCode>),
    pub move_right: (KeyCode, Option<KeyCode>),
    pub soft_drop: (KeyCode, Option<KeyCode>),
    pub hard_drop: (KeyCode, Option<KeyCode>),
    pub rotate_cw: (KeyCode, Option<KeyCode>),
    pub rotate_ccw: (KeyCode, Option<KeyCode>),
    pub hold: (KeyCode, Option<KeyCode>),
    pub pause: (KeyCode, Option<KeyCode>),
}

impl Default for Keybinds {
    /// Mirrors the bindings previously hard-coded in
    /// [`RawKeyboardFrame::from_keyboard`](crate::player::RawKeyboardFrame::from_keyboard):
    /// arrows for move/soft-drop, Space = hard drop, Up/X = rotate CW, Z = rotate
    /// CCW, LeftShift = hold, Escape = pause.
    fn default() -> Self {
        Self {
            move_left: (KeyCode::ArrowLeft, None),
            move_right: (KeyCode::ArrowRight, None),
            soft_drop: (KeyCode::ArrowDown, None),
            hard_drop: (KeyCode::Space, None),
            rotate_cw: (KeyCode::ArrowUp, Some(KeyCode::KeyX)),
            rotate_ccw: (KeyCode::KeyZ, None),
            hold: (KeyCode::ShiftLeft, None),
            pause: (KeyCode::Escape, None),
        }
    }
}

impl Keybinds {
    /// The (primary, secondary) keys bound to `action`.
    pub fn get(&self, action: GameAction) -> (KeyCode, Option<KeyCode>) {
        match action {
            GameAction::MoveLeft => self.move_left,
            GameAction::MoveRight => self.move_right,
            GameAction::SoftDrop => self.soft_drop,
            GameAction::HardDrop => self.hard_drop,
            GameAction::RotateCw => self.rotate_cw,
            GameAction::RotateCcw => self.rotate_ccw,
            GameAction::Hold => self.hold,
            GameAction::Pause => self.pause,
        }
    }

    /// Rebind `action`'s primary key, clearing the secondary. Used by Options.
    pub fn set_primary(&mut self, action: GameAction, key: KeyCode) {
        let slot = match action {
            GameAction::MoveLeft => &mut self.move_left,
            GameAction::MoveRight => &mut self.move_right,
            GameAction::SoftDrop => &mut self.soft_drop,
            GameAction::HardDrop => &mut self.hard_drop,
            GameAction::RotateCw => &mut self.rotate_cw,
            GameAction::RotateCcw => &mut self.rotate_ccw,
            GameAction::Hold => &mut self.hold,
            GameAction::Pause => &mut self.pause,
        };
        *slot = (key, None);
    }
}

/// Player-facing settings. The single shared, mutable contract for tunables.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct GameSettings {
    /// Number of next-piece previews shown (clamped to `MIN_NEXT_COUNT..=MAX_NEXT_COUNT`).
    pub next_count: usize,
    /// Whether the hold mechanic is available.
    pub hold_enabled: bool,
    /// Whether the ghost piece is rendered.
    pub ghost_enabled: bool,
    /// Engine lock-down rule.
    pub lock_down_mode: LockDownMode,
    /// Music volume, 0.0..=1.0.
    pub music_volume: f32,
    /// Sound-effects volume, 0.0..=1.0.
    pub sfx_volume: f32,
    /// Action→key bindings.
    pub keybinds: Keybinds,
}

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            next_count: MAX_NEXT_COUNT,
            hold_enabled: true,
            ghost_enabled: true,
            lock_down_mode: LockDownMode::default(),
            music_volume: 0.5,
            sfx_volume: 0.5,
            keybinds: Keybinds::default(),
        }
    }
}

impl GameSettings {
    /// Clamp every numeric field into its valid range. The options feature calls
    /// this after mutating so out-of-range values never reach gameplay.
    pub fn sanitize(&mut self) {
        self.next_count = self.next_count.clamp(MIN_NEXT_COUNT, MAX_NEXT_COUNT);
        self.music_volume = self.music_volume.clamp(0.0, 1.0);
        self.sfx_volume = self.sfx_volume.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_next_count_is_max_and_within_bounds() {
        let settings = GameSettings::default();
        assert_eq!(settings.next_count, MAX_NEXT_COUNT);
        assert!((MIN_NEXT_COUNT..=MAX_NEXT_COUNT).contains(&settings.next_count));
    }

    #[test]
    fn sanitize_clamps_all_numeric_fields() {
        let mut settings = GameSettings {
            next_count: 99,
            music_volume: 2.0,
            sfx_volume: -1.0,
            ..GameSettings::default()
        };
        settings.sanitize();
        assert_eq!(settings.next_count, MAX_NEXT_COUNT);
        assert_eq!(settings.music_volume, 1.0);
        assert_eq!(settings.sfx_volume, 0.0);
    }

    #[test]
    fn rotate_cw_defaults_to_arrow_up_and_x() {
        let binds = Keybinds::default();
        assert_eq!(binds.get(GameAction::RotateCw), (KeyCode::ArrowUp, Some(KeyCode::KeyX)));
    }

    #[test]
    fn set_primary_replaces_key_and_clears_secondary() {
        let mut binds = Keybinds::default();
        binds.set_primary(GameAction::RotateCw, KeyCode::KeyK);
        assert_eq!(binds.get(GameAction::RotateCw), (KeyCode::KeyK, None));
    }
}
