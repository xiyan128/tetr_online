//! Options feature (A1.7): the interactive settings editor.
//!
//! Builds keyboard-navigable widgets under the [`OptionsRoot`] the screen shell
//! spawns on [`GameState::Options`], letting the player edit
//! [`GameSettings`]: next-piece count (1..=6), hold/ghost toggles, lock-down
//! mode, music/SFX volumes, and the per-action [`Keybinds`]. Every edit calls
//! [`GameSettings::sanitize`] and persists the whole struct through
//! [`StorageResource`] under [`storage::keys::SETTINGS`]; settings are also
//! loaded from there at startup and persisted again on screen exit.
//!
//! Changes take effect because the readers already consume the shared
//! [`GameSettings`] resource: `level_setup` mirrors `next_count` into the
//! previewer/engine, `reconcile_ghost_piece` honors `ghost_enabled`, the engine
//! bridge feeds `lock_down_mode`, and the SFX feature reads the volumes. The
//! keyboard controller reads [`Keybinds`] via [`keyboard_input_from_keybinds`]
//! (see the integrator note in this file's PR report).
//!
//! Encoding: a tiny hand-rolled `key=value` line format (one field per line) so
//! we don't pull in a serializer; see [`encode_settings`]/[`decode_settings`].
//!
//! Touch only this file.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::screens::OptionsRoot;
use crate::settings::{GameAction, GameSettings, Keybinds, MAX_NEXT_COUNT, MIN_NEXT_COUNT};
use crate::storage::{keys, StorageResource};
use crate::ui::focus::{focus_navigation, FocusList, Focusable};
use crate::ui::theme;
use crate::ui::widgets::label_text;
use crate::{engine::LockDownMode, GameState};

/// Options-screen settings editor.
pub struct OptionsPlugin;

impl Plugin for OptionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RebindState>()
            // Load persisted settings once at startup, before any reader uses
            // them. `Startup` runs after `GamePlugin`'s
            // `init_resource::<GameSettings>` / `insert_resource(StorageResource)`,
            // so both resources already exist; the level only reads `GameSettings`
            // on `OnEnter(Playing)`, well after this.
            .add_systems(Startup, load_settings)
            // Persist again when leaving the screen.
            .add_systems(
                OnExit(GameState::Options),
                (clear_rebind_state, save_settings),
            )
            .add_systems(
                Update,
                (
                    // Attach the editor rows once the shell's root exists. Keyed
                    // off `Added<OptionsRoot>` so we never depend on `OnEnter`
                    // system order between this feature and the screen shell.
                    build_options_ui,
                    // While capturing a rebind, swallow nav so arrows/Enter bind
                    // instead of moving focus; otherwise navigate normally.
                    focus_navigation::<OptionsRoot>.run_if(not(rebinding)),
                    edit_options,
                    refresh_option_rows,
                )
                    .chain()
                    .run_if(in_state(GameState::Options)),
            );
    }
}

// ---------------------------------------------------------------------------
// Row model
// ---------------------------------------------------------------------------

/// One editable settings row. Fixed rows come first, then one row per
/// [`GameAction`] rebind, matching their [`Focusable`] indices on the screen.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
enum OptionRow {
    NextCount,
    HoldEnabled,
    GhostEnabled,
    LockDownMode,
    MusicVolume,
    SfxVolume,
    Rebind(GameAction),
}

impl OptionRow {
    /// Fixed (non-rebind) rows, in display order.
    const FIXED: [OptionRow; 6] = [
        OptionRow::NextCount,
        OptionRow::HoldEnabled,
        OptionRow::GhostEnabled,
        OptionRow::LockDownMode,
        OptionRow::MusicVolume,
        OptionRow::SfxVolume,
    ];

    /// Every row in display order (fixed settings then per-action rebinds).
    fn all() -> Vec<OptionRow> {
        OptionRow::FIXED
            .into_iter()
            .chain(GameAction::ALL.into_iter().map(OptionRow::Rebind))
            .collect()
    }

    fn label(self) -> String {
        match self {
            OptionRow::NextCount => "Next Count".into(),
            OptionRow::HoldEnabled => "Hold".into(),
            OptionRow::GhostEnabled => "Ghost Piece".into(),
            OptionRow::LockDownMode => "Lock-Down".into(),
            OptionRow::MusicVolume => "Music Volume".into(),
            OptionRow::SfxVolume => "SFX Volume".into(),
            OptionRow::Rebind(action) => action.label().into(),
        }
    }

    /// The current value rendered on the right of the row.
    fn value(self, settings: &GameSettings, rebind: &RebindState) -> String {
        if let OptionRow::Rebind(action) = self {
            if rebind.capturing == Some(action) {
                return "press a key...".into();
            }
        }
        match self {
            OptionRow::NextCount => settings.next_count.to_string(),
            OptionRow::HoldEnabled => on_off(settings.hold_enabled),
            OptionRow::GhostEnabled => on_off(settings.ghost_enabled),
            OptionRow::LockDownMode => lock_down_label(settings.lock_down_mode).into(),
            OptionRow::MusicVolume => volume_label(settings.music_volume),
            OptionRow::SfxVolume => volume_label(settings.sfx_volume),
            OptionRow::Rebind(action) => key_label(settings.keybinds.get(action).0),
        }
    }
}

fn on_off(value: bool) -> String {
    if value {
        "On".into()
    } else {
        "Off".into()
    }
}

fn volume_label(value: f32) -> String {
    format!("{}%", (value * 100.0).round() as i32)
}

fn lock_down_label(mode: LockDownMode) -> &'static str {
    match mode {
        LockDownMode::Extended => "Extended",
        LockDownMode::Infinite => "Infinite",
        LockDownMode::Classic => "Classic",
    }
}

// ---------------------------------------------------------------------------
// UI markers
// ---------------------------------------------------------------------------

/// Marks the `Text` entity holding a row's right-hand value, so
/// [`refresh_option_rows`] can rewrite it in place.
#[derive(Component)]
struct OptionValueText(OptionRow);

/// Tracks an in-progress keybind capture: while `capturing` is `Some`, the next
/// key press rebinds that action and nav is suppressed.
#[derive(Resource, Default)]
struct RebindState {
    capturing: Option<GameAction>,
}

fn rebinding(state: Res<RebindState>) -> bool {
    state.capturing.is_some()
}

// ---------------------------------------------------------------------------
// Setup: attach editor rows under the shell's OptionsRoot
// ---------------------------------------------------------------------------

fn build_options_ui(
    mut commands: Commands,
    settings: Res<GameSettings>,
    rebind: Res<RebindState>,
    assets: Res<GameAssets>,
    // `Single` skips the system on frames where the root was not just added — the
    // same no-op the early `single()` return used to express.
    root: Single<Entity, Added<OptionsRoot>>,
    existing: Query<(), With<OptionValueText>>,
) {
    let root = *root;
    // Defensive idempotency: never build the rows twice for one screen visit.
    if !existing.is_empty() {
        return;
    }
    let font = assets.font.clone();
    let rows = OptionRow::all();

    // The FocusList lives on the same entity carrying the screen marker the
    // focus helper is generic over (OptionsRoot), per the shared pattern.
    commands.entity(root).insert(FocusList::new(rows.len()));

    // A hint line so the controls are discoverable.
    let hint = commands
        .spawn(label_text(
            "Up/Down select  -  Left/Right adjust  -  Enter toggle/rebind  -  Esc back",
            font.clone(),
        ))
        .id();
    commands.entity(root).add_child(hint);

    for (index, row) in rows.into_iter().enumerate() {
        let value = row.value(&settings, &rebind);
        let entity = commands
            .spawn((
                row,
                Focusable::new(index),
                Node {
                    width: px(320),
                    height: px(30),
                    margin: UiRect::all(px(3)),
                    padding: UiRect::horizontal(px(14)),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(theme::BUTTON_NORMAL),
                children![
                    (
                        Text::new(row.label()),
                        TextFont {
                            font: font.clone(),
                            font_size: theme::BUTTON_FONT_SIZE,
                            ..default()
                        },
                        TextColor(theme::TEXT),
                    ),
                    (
                        OptionValueText(row),
                        Text::new(value),
                        TextFont {
                            font: font.clone(),
                            font_size: theme::BUTTON_FONT_SIZE,
                            ..default()
                        },
                        TextColor(theme::ACCENT),
                    ),
                ],
            ))
            .id();
        commands.entity(root).add_child(entity);
    }
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

/// Handle input against the focused row. Left/Right adjust numeric & enum
/// settings, Enter toggles bools / cycles lock-down / starts a rebind (or, while
/// capturing, the pressed key becomes the new binding). Persists after any
/// change.
///
/// Esc is owned by the screen shell (`screens/options.rs`), which exits to the
/// main menu; this system never sets state. While capturing a rebind, Esc
/// cancels the capture (no key is bound) — the shell still exits on that same
/// Esc, which is the intuitive "get me out" behavior. Esc therefore can't be
/// *bound* through the UI (it stays reserved for back/pause).
fn edit_options(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<GameSettings>,
    mut rebind: ResMut<RebindState>,
    // Stays a plain `Query` (not `Single`): the rebind-capture branch below must
    // run even on a frame with no/zero focus list, so the system can't be skipped.
    lists: Query<&FocusList, With<OptionsRoot>>,
    rows: Query<(&Focusable, &OptionRow)>,
    storage: Res<StorageResource>,
) {
    // --- Rebind capture takes priority over everything else. ---
    if let Some(action) = rebind.capturing {
        if keys.just_pressed(KeyCode::Escape) {
            rebind.capturing = None;
            return;
        }
        if let Some(key) = first_just_pressed(&keys) {
            settings.keybinds.set_primary(action, key);
            settings.sanitize();
            persist(&storage, &settings);
            rebind.capturing = None;
        }
        return;
    }

    let Ok(list) = lists.single() else {
        return;
    };

    let focused = rows
        .iter()
        .find(|(f, _)| f.index == list.index)
        .map(|(_, row)| *row);
    let Some(row) = focused else {
        return;
    };

    let left = keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::KeyA);
    let right = keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::KeyD);
    let activate = keys.just_pressed(KeyCode::Enter)
        || keys.just_pressed(KeyCode::NumpadEnter)
        || keys.just_pressed(KeyCode::Space);

    let mut changed = false;
    match row {
        OptionRow::NextCount => {
            if right {
                settings.next_count = settings.next_count.saturating_add(1);
                changed = true;
            } else if left {
                settings.next_count = settings.next_count.saturating_sub(1);
                changed = true;
            }
        }
        OptionRow::HoldEnabled => {
            if left || right || activate {
                settings.hold_enabled = !settings.hold_enabled;
                changed = true;
            }
        }
        OptionRow::GhostEnabled => {
            if left || right || activate {
                settings.ghost_enabled = !settings.ghost_enabled;
                changed = true;
            }
        }
        OptionRow::LockDownMode => {
            if right || activate {
                settings.lock_down_mode = cycle_lock_down(settings.lock_down_mode, 1);
                changed = true;
            } else if left {
                settings.lock_down_mode = cycle_lock_down(settings.lock_down_mode, -1);
                changed = true;
            }
        }
        OptionRow::MusicVolume => {
            if right {
                settings.music_volume += VOLUME_STEP;
                changed = true;
            } else if left {
                settings.music_volume -= VOLUME_STEP;
                changed = true;
            }
        }
        OptionRow::SfxVolume => {
            if right {
                settings.sfx_volume += VOLUME_STEP;
                changed = true;
            } else if left {
                settings.sfx_volume -= VOLUME_STEP;
                changed = true;
            }
        }
        OptionRow::Rebind(action) => {
            if activate {
                rebind.capturing = Some(action);
            }
        }
    }

    if changed {
        // `next_count` must wrap (per spec: 1..=6 cycles); volumes are clamped by
        // sanitize. Handle the wrap explicitly before sanitize clamps it.
        if matches!(row, OptionRow::NextCount) {
            if settings.next_count > MAX_NEXT_COUNT {
                settings.next_count = MIN_NEXT_COUNT;
            } else if settings.next_count < MIN_NEXT_COUNT {
                settings.next_count = MAX_NEXT_COUNT;
            }
        }
        settings.sanitize();
        persist(&storage, &settings);
    }
}

const VOLUME_STEP: f32 = 0.1;

fn cycle_lock_down(mode: LockDownMode, delta: i32) -> LockDownMode {
    const ORDER: [LockDownMode; 3] = [
        LockDownMode::Extended,
        LockDownMode::Infinite,
        LockDownMode::Classic,
    ];
    let idx = ORDER.iter().position(|&m| m == mode).unwrap_or(0) as i32;
    let next = (idx + delta).rem_euclid(ORDER.len() as i32) as usize;
    ORDER[next]
}

/// The first key that transitioned to pressed this frame, ignoring nothing —
/// used to capture a rebind. Returns `None` if no key was just pressed.
fn first_just_pressed(keys: &ButtonInput<KeyCode>) -> Option<KeyCode> {
    keys.get_just_pressed().next().copied()
}

/// Rewrite each row's value text so the UI reflects the current settings (after
/// an edit) and the "press a key..." prompt while capturing.
fn refresh_option_rows(
    settings: Res<GameSettings>,
    rebind: Res<RebindState>,
    mut texts: Query<(&OptionValueText, &mut Text)>,
) {
    if !settings.is_changed() && !rebind.is_changed() {
        return;
    }
    for (marker, mut text) in &mut texts {
        text.0 = marker.0.value(&settings, &rebind);
    }
}

fn clear_rebind_state(mut rebind: ResMut<RebindState>) {
    rebind.capturing = None;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

fn load_settings(storage: Res<StorageResource>, mut settings: ResMut<GameSettings>) {
    if let Some(raw) = storage.0.load(keys::SETTINGS) {
        if let Some(loaded) = decode_settings(&raw) {
            *settings = loaded;
            settings.sanitize();
        }
    }
}

fn save_settings(storage: Res<StorageResource>, settings: Res<GameSettings>) {
    persist(&storage, &settings);
}

fn persist(storage: &StorageResource, settings: &GameSettings) {
    storage.0.save(keys::SETTINGS, &encode_settings(settings));
}

/// Serialize [`GameSettings`] to a `key=value`-per-line blob. Stable, forgiving
/// format: unknown lines are ignored on decode and any missing field falls back
/// to its [`Default`], so older/newer blobs degrade gracefully.
fn encode_settings(s: &GameSettings) -> String {
    let mut out = String::new();
    out.push_str(&format!("next_count={}\n", s.next_count));
    out.push_str(&format!("hold_enabled={}\n", s.hold_enabled));
    out.push_str(&format!("ghost_enabled={}\n", s.ghost_enabled));
    out.push_str(&format!(
        "lock_down_mode={}\n",
        lock_down_token(s.lock_down_mode)
    ));
    out.push_str(&format!("music_volume={}\n", s.music_volume));
    out.push_str(&format!("sfx_volume={}\n", s.sfx_volume));
    for action in GameAction::ALL {
        let (primary, secondary) = s.keybinds.get(action);
        // Persist primary (+ optional secondary) as key codes.
        let sec = secondary
            .and_then(key_code_token)
            .map(|t| format!(",{t}"))
            .unwrap_or_default();
        if let Some(prim) = key_code_token(primary) {
            out.push_str(&format!("bind.{}={}{}\n", action_token(action), prim, sec));
        }
    }
    out
}

/// Parse a blob produced by [`encode_settings`]. Returns `None` only if the
/// input is empty/whitespace; otherwise builds on top of [`GameSettings::default`]
/// so partial blobs still load.
fn decode_settings(raw: &str) -> Option<GameSettings> {
    if raw.trim().is_empty() {
        return None;
    }
    let mut s = GameSettings::default();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "next_count" => {
                if let Ok(v) = value.parse() {
                    s.next_count = v;
                }
            }
            "hold_enabled" => {
                if let Ok(v) = value.parse() {
                    s.hold_enabled = v;
                }
            }
            "ghost_enabled" => {
                if let Ok(v) = value.parse() {
                    s.ghost_enabled = v;
                }
            }
            "lock_down_mode" => {
                if let Some(v) = lock_down_from_token(value) {
                    s.lock_down_mode = v;
                }
            }
            "music_volume" => {
                if let Ok(v) = value.parse() {
                    s.music_volume = v;
                }
            }
            "sfx_volume" => {
                if let Ok(v) = value.parse() {
                    s.sfx_volume = v;
                }
            }
            other => {
                if let Some(action_name) = other.strip_prefix("bind.") {
                    if let Some(action) = action_from_token(action_name) {
                        // value = "PRIMARY" or "PRIMARY,SECONDARY". Restore the
                        // full tuple so default aliases (e.g. rotate-CW = Up,X)
                        // survive a round-trip; the rebind UI itself only ever
                        // sets a primary (clearing the secondary) via set_primary.
                        let mut parts = value.split(',');
                        let primary = parts.next().and_then(key_code_from_token);
                        let secondary = parts.next().and_then(key_code_from_token);
                        if let Some(primary) = primary {
                            apply_binding(&mut s.keybinds, action, (primary, secondary));
                        }
                    }
                }
            }
        }
    }
    Some(s)
}

fn lock_down_token(mode: LockDownMode) -> &'static str {
    match mode {
        LockDownMode::Extended => "extended",
        LockDownMode::Infinite => "infinite",
        LockDownMode::Classic => "classic",
    }
}

fn lock_down_from_token(token: &str) -> Option<LockDownMode> {
    match token {
        "extended" => Some(LockDownMode::Extended),
        "infinite" => Some(LockDownMode::Infinite),
        "classic" => Some(LockDownMode::Classic),
        _ => None,
    }
}

fn action_token(action: GameAction) -> &'static str {
    match action {
        GameAction::MoveLeft => "move_left",
        GameAction::MoveRight => "move_right",
        GameAction::SoftDrop => "soft_drop",
        GameAction::HardDrop => "hard_drop",
        GameAction::RotateCw => "rotate_cw",
        GameAction::RotateCcw => "rotate_ccw",
        GameAction::Hold => "hold",
        GameAction::Pause => "pause",
    }
}

fn action_from_token(token: &str) -> Option<GameAction> {
    GameAction::ALL
        .into_iter()
        .find(|a| action_token(*a) == token)
}

/// Write a full `(primary, secondary)` binding for `action`. `Keybinds` only
/// exposes `set_primary` (which clears the secondary), but its fields are public;
/// persistence uses this so a stored secondary alias is restored exactly. The
/// rebind UI still goes through `set_primary`.
fn apply_binding(binds: &mut Keybinds, action: GameAction, value: (KeyCode, Option<KeyCode>)) {
    let slot = match action {
        GameAction::MoveLeft => &mut binds.move_left,
        GameAction::MoveRight => &mut binds.move_right,
        GameAction::SoftDrop => &mut binds.soft_drop,
        GameAction::HardDrop => &mut binds.hard_drop,
        GameAction::RotateCw => &mut binds.rotate_cw,
        GameAction::RotateCcw => &mut binds.rotate_ccw,
        GameAction::Hold => &mut binds.hold,
        GameAction::Pause => &mut binds.pause,
    };
    *slot = value;
}

// ---------------------------------------------------------------------------
// KeyCode <-> string mapping
// ---------------------------------------------------------------------------
//
// Bevy's `KeyCode` has no stable string parse, so we keep an explicit table for
// the keys a player can plausibly bind. Serialization uses the same table; an
// unmapped key simply isn't persisted (and the display falls back to Debug).

/// Stable token for persistence, or `None` if we don't have a round-trippable
/// name for this key (it then won't be saved, preserving the prior binding).
fn key_code_token(code: KeyCode) -> Option<&'static str> {
    KEY_TABLE
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, token)| *token)
}

fn key_code_from_token(token: &str) -> Option<KeyCode> {
    KEY_TABLE
        .iter()
        .find(|(_, t)| *t == token)
        .map(|(code, _)| *code)
}

/// Short human-facing label for a bound key (shown in the rebind rows). Falls
/// back to the `Debug` name for anything outside the table.
fn key_label(code: KeyCode) -> String {
    if let Some(token) = key_code_token(code) {
        // Tokens double as compact labels (uppercased for letters/arrows).
        return token.to_string();
    }
    format!("{code:?}")
}

/// The bindable-key table: `(KeyCode, token)`. Tokens are also the on-screen
/// labels. Covers letters, digits, arrows, and the common modifier/whitespace
/// keys — enough for any of the eight actions.
#[rustfmt::skip]
const KEY_TABLE: &[(KeyCode, &str)] = &[
    (KeyCode::ArrowLeft, "Left"), (KeyCode::ArrowRight, "Right"),
    (KeyCode::ArrowUp, "Up"), (KeyCode::ArrowDown, "Down"),
    (KeyCode::Space, "Space"), (KeyCode::Enter, "Enter"),
    (KeyCode::Escape, "Esc"), (KeyCode::Tab, "Tab"),
    (KeyCode::ShiftLeft, "LShift"), (KeyCode::ShiftRight, "RShift"),
    (KeyCode::ControlLeft, "LCtrl"), (KeyCode::ControlRight, "RCtrl"),
    (KeyCode::AltLeft, "LAlt"), (KeyCode::AltRight, "RAlt"),
    (KeyCode::Comma, ","), (KeyCode::Period, "."), (KeyCode::Slash, "/"),
    (KeyCode::Semicolon, ";"), (KeyCode::Quote, "'"),
    (KeyCode::BracketLeft, "["), (KeyCode::BracketRight, "]"),
    (KeyCode::KeyA, "A"), (KeyCode::KeyB, "B"), (KeyCode::KeyC, "C"),
    (KeyCode::KeyD, "D"), (KeyCode::KeyE, "E"), (KeyCode::KeyF, "F"),
    (KeyCode::KeyG, "G"), (KeyCode::KeyH, "H"), (KeyCode::KeyI, "I"),
    (KeyCode::KeyJ, "J"), (KeyCode::KeyK, "K"), (KeyCode::KeyL, "L"),
    (KeyCode::KeyM, "M"), (KeyCode::KeyN, "N"), (KeyCode::KeyO, "O"),
    (KeyCode::KeyP, "P"), (KeyCode::KeyQ, "Q"), (KeyCode::KeyR, "R"),
    (KeyCode::KeyS, "S"), (KeyCode::KeyT, "T"), (KeyCode::KeyU, "U"),
    (KeyCode::KeyV, "V"), (KeyCode::KeyW, "W"), (KeyCode::KeyX, "X"),
    (KeyCode::KeyY, "Y"), (KeyCode::KeyZ, "Z"),
    (KeyCode::Digit0, "0"), (KeyCode::Digit1, "1"), (KeyCode::Digit2, "2"),
    (KeyCode::Digit3, "3"), (KeyCode::Digit4, "4"), (KeyCode::Digit5, "5"),
    (KeyCode::Digit6, "6"), (KeyCode::Digit7, "7"), (KeyCode::Digit8, "8"),
    (KeyCode::Digit9, "9"),
];

// ---------------------------------------------------------------------------
// Keybind read-path for the controller (migration helper)
// ---------------------------------------------------------------------------

/// Build raw per-frame [`RawKeyboardFrame`](crate::player::RawKeyboardFrame) from
/// Bevy's keyboard state using the player's [`Keybinds`], replacing the
/// hard-coded mapping in `RawKeyboardFrame::from_keyboard`.
///
/// This is the read-path the gameplay driver should call so remapped keys take
/// effect. The integrator wires it at the single call site in
/// `src/level/mod.rs` (see this PR's report) since that file is outside the
/// options feature's ownership.
pub fn keyboard_input_from_keybinds(
    keyboard: &ButtonInput<KeyCode>,
    binds: &Keybinds,
    dt_seconds: f32,
) -> crate::player::RawKeyboardFrame {
    use crate::player::RawKeyboardFrame;

    // `pressed`/`just_pressed` against either bound key for an action.
    let pressed = |action: GameAction| {
        let (primary, secondary) = binds.get(action);
        keyboard.pressed(primary) || secondary.is_some_and(|k| keyboard.pressed(k))
    };
    let just = |action: GameAction| {
        let (primary, secondary) = binds.get(action);
        keyboard.just_pressed(primary) || secondary.is_some_and(|k| keyboard.just_pressed(k))
    };

    RawKeyboardFrame {
        dt_seconds,
        left_pressed: pressed(GameAction::MoveLeft),
        right_pressed: pressed(GameAction::MoveRight),
        left_just_pressed: just(GameAction::MoveLeft),
        right_just_pressed: just(GameAction::MoveRight),
        soft_drop: pressed(GameAction::SoftDrop),
        hard_drop_just_pressed: just(GameAction::HardDrop),
        rotate_cw_just_pressed: just(GameAction::RotateCw),
        rotate_ccw_just_pressed: just(GameAction::RotateCcw),
        hold_just_pressed: just(GameAction::Hold),
        pause_just_pressed: just(GameAction::Pause),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_default_settings() {
        let settings = GameSettings::default();
        let encoded = encode_settings(&settings);
        let decoded = decode_settings(&encoded).expect("non-empty blob decodes");
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

        let decoded = decode_settings(&encode_settings(&settings)).unwrap();
        assert_eq!(decoded, settings);
    }

    #[test]
    fn empty_blob_yields_none() {
        assert!(decode_settings("").is_none());
        assert!(decode_settings("   \n  ").is_none());
    }

    #[test]
    fn partial_and_garbage_blob_falls_back_to_defaults() {
        // Only one field set; unknown lines ignored; result is defaults + override.
        let decoded = decode_settings("ghost_enabled=false\nnonsense\nbogus=key\n").unwrap();
        let expected = GameSettings {
            ghost_enabled: false,
            ..GameSettings::default()
        };
        assert_eq!(decoded, expected);
    }

    #[test]
    fn decode_clamps_via_caller_sanitize_contract() {
        // decode itself is lenient; sanitize (called by load_settings) clamps.
        let mut decoded = decode_settings("next_count=99\nmusic_volume=5.0\n").unwrap();
        decoded.sanitize();
        assert_eq!(decoded.next_count, MAX_NEXT_COUNT);
        assert_eq!(decoded.music_volume, 1.0);
    }

    #[test]
    fn key_table_round_trips_every_entry() {
        for (code, token) in KEY_TABLE {
            assert_eq!(key_code_token(*code), Some(*token));
            assert_eq!(key_code_from_token(token), Some(*code));
        }
    }

    #[test]
    fn lock_down_cycles_forward_and_back() {
        assert_eq!(
            cycle_lock_down(LockDownMode::Extended, 1),
            LockDownMode::Infinite
        );
        assert_eq!(
            cycle_lock_down(LockDownMode::Classic, 1),
            LockDownMode::Extended
        );
        assert_eq!(
            cycle_lock_down(LockDownMode::Extended, -1),
            LockDownMode::Classic
        );
    }

    #[test]
    fn all_rows_cover_fixed_plus_every_action() {
        let rows = OptionRow::all();
        assert_eq!(rows.len(), OptionRow::FIXED.len() + GameAction::ALL.len());
        for action in GameAction::ALL {
            assert!(rows.contains(&OptionRow::Rebind(action)));
        }
    }

    #[test]
    fn every_default_keybind_is_persistable() {
        // If any default key lacked a KEY_TABLE entry it would be silently
        // dropped on save, so guard against that explicitly.
        let binds = Keybinds::default();
        for action in GameAction::ALL {
            let (primary, secondary) = binds.get(action);
            assert!(
                key_code_token(primary).is_some(),
                "{action:?} primary not in table"
            );
            if let Some(sec) = secondary {
                assert!(
                    key_code_token(sec).is_some(),
                    "{action:?} secondary not in table"
                );
            }
        }
    }

    #[test]
    fn keybinds_read_path_honors_primary_secondary_and_remaps() {
        let mut binds = Keybinds::default();
        // Remap hard-drop off Space onto KeyJ.
        binds.set_primary(GameAction::HardDrop, KeyCode::KeyJ);

        let mut keyboard = ButtonInput::<KeyCode>::default();
        keyboard.press(KeyCode::KeyX); // rotate-CW secondary alias
        keyboard.press(KeyCode::KeyJ); // remapped hard drop
        keyboard.press(KeyCode::ArrowLeft); // move-left primary

        let input = keyboard_input_from_keybinds(&keyboard, &binds, 0.016);
        assert!(
            input.rotate_cw_just_pressed,
            "secondary alias should trigger rotate CW"
        );
        assert!(
            input.hard_drop_just_pressed,
            "remapped key should trigger hard drop"
        );
        assert!(input.left_pressed && input.left_just_pressed);
        assert!(!input.soft_drop, "unpressed action stays false");
        assert_eq!(input.dt_seconds, 0.016);
    }
}
