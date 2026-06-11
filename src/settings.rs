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
//! It also owns its own persistence: [`encode_settings`]/[`decode_settings`]
//! round-trip it through RON via `serde` derives, so the storage format is a
//! property of the settings type, not of the options UI. The options feature
//! drives the read/write *systems* and renders the editor; the wire *format*
//! lives here, next to the data it serializes.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::engine::LockDownMode;

/// The smallest and largest number of next-piece previews the UI/engine support.
pub const MIN_NEXT_COUNT: usize = 1;
pub const MAX_NEXT_COUNT: usize = 6;

/// A logical, rebindable player action. The keyboard controller maps the bound
/// [`KeyCode`]s to its raw input each frame; the Options screen rebinds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
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
/// (e.g. rotate-CW defaults to both `ArrowUp` and `KeyW`). The keyboard
/// controller reads `pressed`/`just_pressed` for *either* key.
#[derive(Resource, Debug, Clone, PartialEq, Eq, Reflect, Serialize, Deserialize)]
#[reflect(Resource)]
// A missing binding falls back to its default, so older/partial blobs still load.
// `KeyCode` serializes via Bevy's own derive (the `serialize` feature), so a key is
// stored under its stable Bevy variant name.
#[serde(default)]
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
    /// Arrows OR WASD out of the box: arrows/AD for movement, Down/S soft
    /// drop, Up/W rotate CW (W mirrors ArrowUp, displacing the old redundant
    /// X alias), Z = rotate CCW, Space = hard drop, LeftShift = hold,
    /// Escape = pause. Both hand positions work without a trip to Options.
    fn default() -> Self {
        Self {
            move_left: (KeyCode::ArrowLeft, Some(KeyCode::KeyA)),
            move_right: (KeyCode::ArrowRight, Some(KeyCode::KeyD)),
            soft_drop: (KeyCode::ArrowDown, Some(KeyCode::KeyS)),
            hard_drop: (KeyCode::Space, None),
            rotate_cw: (KeyCode::ArrowUp, Some(KeyCode::KeyW)),
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
#[derive(Resource, Debug, Clone, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect(Resource)]
// Missing fields fall back to `Default` and unknown fields are ignored, so a
// partial, older, or newer settings blob still loads (see `decode_settings`).
#[serde(default)]
pub struct GameSettings {
    /// Number of next-piece previews shown (clamped to `MIN_NEXT_COUNT..=MAX_NEXT_COUNT`).
    pub next_count: usize,
    /// Whether the hold mechanic is available.
    pub hold_enabled: bool,
    /// Whether the ghost piece is rendered.
    pub ghost_enabled: bool,
    /// Engine lock-down rule. `LockDownMode` lives in the engine-agnostic
    /// `engine/` crate, which depends on neither Bevy nor `serde`; it is skipped
    /// for reflection and (de)serialized via the [`lock_down_serde`] token adapter
    /// rather than coupling the engine to either framework.
    #[reflect(ignore)]
    #[serde(with = "lock_down_serde")]
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

// ---------------------------------------------------------------------------
// Persistence codec (RON)
// ---------------------------------------------------------------------------

/// Serialize [`GameSettings`] to a pretty-printed RON blob for storage.
///
/// Serialization of this type is infallible (nothing in the graph has a fallible
/// `Serialize`), so a failure could only be a logic bug; we degrade to an empty
/// string rather than panic, which simply reloads as defaults.
pub(crate) fn encode_settings(settings: &GameSettings) -> String {
    ron::ser::to_string_pretty(settings, ron::ser::PrettyConfig::default()).unwrap_or_default()
}

/// Parse a RON blob produced by [`encode_settings`]. Returns `None` for empty or
/// unparseable input (the caller keeps the in-memory defaults). Missing fields
/// fall back to their `Default` and unknown fields are ignored — both via the
/// `#[serde(default)]` on [`GameSettings`]/[`Keybinds`] — so partial, older, or
/// newer blobs degrade gracefully instead of failing the load.
pub(crate) fn decode_settings(raw: &str) -> Option<GameSettings> {
    ron::from_str(raw).ok()
}

/// `serde` adapter for [`LockDownMode`], an engine enum kept free of both Bevy
/// `Reflect` and `serde` so the pure rules core depends on neither framework. It
/// (de)serializes as a stable lowercase token; an unrecognized token degrades to
/// the engine default rather than failing the whole settings load.
mod lock_down_serde {
    use super::LockDownMode;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(mode: &LockDownMode, s: S) -> Result<S::Ok, S::Error> {
        let token = match mode {
            LockDownMode::Extended => "extended",
            LockDownMode::Infinite => "infinite",
            LockDownMode::Classic => "classic",
        };
        token.serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<LockDownMode, D::Error> {
        let token = String::deserialize(d)?;
        Ok(match token.as_str() {
            "extended" => LockDownMode::Extended,
            "infinite" => LockDownMode::Infinite,
            "classic" => LockDownMode::Classic,
            _ => LockDownMode::default(), // unknown token -> the engine default
        })
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
    fn the_default_map_covers_arrows_and_wasd() {
        let binds = Keybinds::default();
        assert_eq!(
            binds.get(GameAction::RotateCw),
            (KeyCode::ArrowUp, Some(KeyCode::KeyW)),
            "W mirrors ArrowUp so a WASD hand can rotate"
        );
        assert_eq!(
            binds.get(GameAction::MoveLeft),
            (KeyCode::ArrowLeft, Some(KeyCode::KeyA))
        );
        assert_eq!(
            binds.get(GameAction::MoveRight),
            (KeyCode::ArrowRight, Some(KeyCode::KeyD))
        );
        assert_eq!(
            binds.get(GameAction::SoftDrop),
            (KeyCode::ArrowDown, Some(KeyCode::KeyS))
        );
    }

    #[test]
    fn set_primary_replaces_key_and_clears_secondary() {
        let mut binds = Keybinds::default();
        binds.set_primary(GameAction::RotateCw, KeyCode::KeyK);
        assert_eq!(binds.get(GameAction::RotateCw), (KeyCode::KeyK, None));
    }

    #[test]
    fn round_trips_default_settings() {
        let settings = GameSettings::default();
        let decoded = decode_settings(&encode_settings(&settings)).expect("blob decodes");
        assert_eq!(decoded, settings);
    }

    #[test]
    fn round_trips_edited_settings() {
        let mut settings = GameSettings {
            next_count: 3,
            hold_enabled: false,
            ghost_enabled: false,
            lock_down_mode: LockDownMode::Classic,
            music_volume: 0.2,
            sfx_volume: 0.9,
            ..GameSettings::default()
        };
        settings
            .keybinds
            .set_primary(GameAction::HardDrop, KeyCode::KeyK);
        settings
            .keybinds
            .set_primary(GameAction::MoveLeft, KeyCode::KeyA);

        let decoded = decode_settings(&encode_settings(&settings)).expect("blob decodes");
        assert_eq!(decoded, settings);
    }

    #[test]
    fn secondary_keybind_alias_survives_a_round_trip() {
        // The default rotate-CW binds a secondary alias (W); the round trip must
        // keep the full (primary, secondary) tuple, not just the primary.
        let decoded = decode_settings(&encode_settings(&GameSettings::default())).unwrap();
        assert_eq!(
            decoded.keybinds.get(GameAction::RotateCw),
            (KeyCode::ArrowUp, Some(KeyCode::KeyW))
        );
    }

    #[test]
    fn lock_down_mode_round_trips_every_variant() {
        for mode in [
            LockDownMode::Extended,
            LockDownMode::Infinite,
            LockDownMode::Classic,
        ] {
            let settings = GameSettings {
                lock_down_mode: mode,
                ..GameSettings::default()
            };
            let decoded = decode_settings(&encode_settings(&settings)).unwrap();
            assert_eq!(decoded.lock_down_mode, mode);
        }
    }

    #[test]
    fn empty_or_garbage_blob_yields_none() {
        assert!(decode_settings("").is_none());
        assert!(decode_settings("   \n  ").is_none());
        assert!(decode_settings("this is not ron {{{").is_none());
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // A partial RON blob overrides one field; the rest come from `Default`.
        let decoded = decode_settings("(ghost_enabled: false)").expect("partial blob decodes");
        let expected = GameSettings {
            ghost_enabled: false,
            ..GameSettings::default()
        };
        assert_eq!(decoded, expected);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // A field this version doesn't know must not fail the load (forward compat).
        let decoded = decode_settings("(next_count: 4, some_future_field: 7)")
            .expect("unknown fields are skipped");
        assert_eq!(decoded.next_count, 4);
    }

    #[test]
    fn decode_is_lenient_and_sanitize_clamps() {
        // decode itself is lenient; sanitize (called by load_settings) clamps.
        let mut decoded = decode_settings("(next_count: 99, music_volume: 5.0)").unwrap();
        decoded.sanitize();
        assert_eq!(decoded.next_count, MAX_NEXT_COUNT);
        assert_eq!(decoded.music_volume, 1.0);
    }
}
