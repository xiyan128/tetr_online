//! Keyboard focus-navigation helper for menu screens.
//!
//! A screen marks each selectable row with [`Focusable`] (carrying its 0-based
//! index) and puts one [`FocusList`] on the screen root tracking how many items
//! there are and which is focused. [`focus_navigation`] (generic over a screen
//! marker `M`) reads Up/Down to move the cursor and restyles the focused button.
//! [`focus_activation`] reports Enter (select) and Esc (back) as events the
//! screen reacts to.
//!
//! Screen plugins (and the options/help/high-scores feature agents) reuse this
//! so every menu navigates identically.

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;

use super::theme;

/// A selectable menu row. `index` is its position within the screen's
/// [`FocusList`] (0-based, top to bottom).
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct Focusable {
    pub index: usize,
}

/// Marks a row's label text as focus-restyled (cream at rest, amber on focus,
/// ground-on-amber while pressed). Texts WITHOUT this marker keep their own
/// color — the options screen's amber value column stays amber regardless of
/// where the cursor is.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct FocusLabel;

impl Focusable {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

/// Per-screen focus cursor. Place one on the screen root entity. `count` is the
/// number of [`Focusable`] rows; `index` is the focused one.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct FocusList {
    pub index: usize,
    pub count: usize,
}

impl FocusList {
    pub fn new(count: usize) -> Self {
        Self { index: 0, count }
    }

    /// Move the cursor by `delta` rows, wrapping. No-op when there are no rows.
    pub fn move_by(&mut self, delta: isize) {
        if self.count == 0 {
            return;
        }
        let count = self.count as isize;
        let next = (self.index as isize + delta).rem_euclid(count);
        self.index = next as usize;
    }
}

/// The per-row query data the focus restyle works over: identity, optional
/// pointer state, the two restyled surfaces, and the (optional) label
/// children. A row participates only if it carries BOTH color components.
type FocusRowItem = (
    &'static Focusable,
    Option<&'static Interaction>,
    &'static mut BackgroundColor,
    &'static mut BorderColor,
    Option<&'static Children>,
);

/// What the player did on a focused menu this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavAction {
    /// Enter/Space pressed — activate the focused row (carries its index).
    Select(usize),
    /// Esc pressed — go back / up one screen.
    Back,
}

/// Drive focus from BOTH keyboard and mouse, and restyle the focusable buttons.
/// Generic over a screen marker `M` so each screen's lists are isolated.
///
/// - **Mouse:** moving the pointer onto a row (or pressing one) moves the cursor
///   to it, so the keyboard focus and the pointer agree and a click lands on the
///   highlighted item. A pressed row shows the pressed color.
/// - **Keyboard:** Up/Down (and W/S) move the cursor.
///
/// "Most recent input device wins": a hover only claims the cursor on a frame the
/// pointer actually *moved*. A pointer merely resting over a row must not re-grab
/// focus every frame, or it would immediately undo an arrow-key press (the row
/// stays `Hovered`, so the cursor would snap back under the pointer on the very
/// next frame) — that was the "hover + arrow keys fight" bug.
///
/// `M` is the screen-root marker component; the [`FocusList`] is expected on the
/// same entity. Call `app.add_systems(Update, focus_navigation::<MyScreen>.run_if(in_state(...)))`.
/// `Single` makes the scheduler skip this unless exactly one `FocusList` with
/// marker `M` exists — the only state in which the rows are meaningful.
pub fn focus_navigation<M: Component>(
    keys: Res<ButtonInput<KeyCode>>,
    pointer_motion: Res<AccumulatedMouseMotion>,
    mut list: Single<&mut FocusList, With<M>>,
    mut buttons: Query<FocusRowItem>,
    mut labels: Query<&mut TextColor, With<FocusLabel>>,
) {
    // Move the cursor to the pointed row — but weigh a press and a hover
    // differently. A press is an explicit action, so it always claims the cursor;
    // a hover only does on a frame the pointer moved (see the doc comment).
    // `Interaction` is optional: keyboard-only rows (the options list) still
    // restyle, they just never claim the cursor by pointer.
    let pointer_moved = pointer_motion.delta != Vec2::ZERO;
    if let Some(index) = buttons
        .iter()
        .find_map(|(focusable, interaction, _, _, _)| {
            let claims_cursor = match interaction.copied().unwrap_or(Interaction::None) {
                Interaction::Pressed => true,
                Interaction::Hovered => pointer_moved,
                Interaction::None => false,
            };
            claims_cursor.then_some(focusable.index)
        })
    {
        list.index = index;
    }

    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        list.move_by(1);
    }
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        list.move_by(-1);
    }

    // Kissaten button states: a resting row is ground + frame border; focus
    // turns the border and label amber; a press inverts into an amber chip.
    // NOTE: a row participates only if it carries BackgroundColor AND
    // BorderColor (every current row builder does); Interaction and Children
    // are optional so keyboard-only or childless rows still restyle.
    for (focusable, interaction, mut bg, mut border, children) in &mut buttons {
        let pressed = interaction.copied() == Some(Interaction::Pressed);
        let (bg_color, border_color, text_color) = if pressed {
            (theme::ACCENT, theme::ACCENT, theme::BG)
        } else if focusable.index == list.index {
            (theme::BG, theme::ACCENT, theme::ACCENT)
        } else {
            (theme::BG, theme::FRAME, theme::TEXT)
        };
        *bg = bg_color.into();
        *border = BorderColor::all(border_color);
        for child in children.into_iter().flatten() {
            if let Ok(mut label) = labels.get_mut(*child) {
                label.0 = text_color;
            }
        }
    }
}

/// Detect Enter/Space (select focused) and Esc (back), returning a [`NavAction`]
/// for the caller to handle against the current [`FocusList`]. Keyboard only;
/// pair with [`clicked_focusable`] for mouse selection.
///
/// Screens typically wrap this in their own system that matches on the result
/// and sets `NextState<GameState>` accordingly.
pub fn read_nav_action(keys: &ButtonInput<KeyCode>, list: &FocusList) -> Option<NavAction> {
    if keys.just_pressed(KeyCode::Escape) {
        return Some(NavAction::Back);
    }
    if keys.just_pressed(KeyCode::Enter)
        || keys.just_pressed(KeyCode::NumpadEnter)
        || keys.just_pressed(KeyCode::Space)
    {
        return Some(NavAction::Select(list.index));
    }
    None
}

/// The 0-based index of a focusable menu button clicked **this frame** — the
/// mouse counterpart to [`read_nav_action`]'s `Select`.
///
/// A screen's activation handler treats this exactly like a keyboard Select:
/// `read_nav_action(..).or_else(|| clicked_focusable(&clicks).map(NavAction::Select))`.
/// [`focus_navigation`] already moves the cursor to the hovered row, so the click
/// and the highlight always agree.
///
/// The query is `Changed<Interaction>`-filtered, so this is **edge-triggered**:
/// it reports the press once, on the frame `Interaction` becomes `Pressed`.
/// `Interaction` is a level (it stays `Pressed` while the button is held), and
/// most screens were masked from that only because their Select action left
/// the screen — a Select that *stays* on the screen (the versus seat pickers,
/// Rematch) would otherwise re-fire every frame of one physical click.
pub fn clicked_focusable(
    buttons: &Query<(&Focusable, &Interaction), Changed<Interaction>>,
) -> Option<usize> {
    buttons.iter().find_map(|(focusable, interaction)| {
        (*interaction == Interaction::Pressed).then_some(focusable.index)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_by_wraps_both_directions() {
        let mut list = FocusList::new(3);
        list.move_by(1);
        assert_eq!(list.index, 1);
        list.move_by(-1);
        assert_eq!(list.index, 0);
        // Wrap past the top.
        list.move_by(-1);
        assert_eq!(list.index, 2);
        // Wrap past the bottom.
        list.move_by(1);
        assert_eq!(list.index, 0);
    }

    #[test]
    fn move_by_is_noop_with_no_items() {
        let mut list = FocusList::new(0);
        list.move_by(1);
        assert_eq!(list.index, 0);
    }

    /// A physical click holds `Interaction::Pressed` for several frames; a
    /// Select that *stays on its screen* (the versus seat picker, Rematch)
    /// must fire once per click, not once per frame — the `Changed` filter on
    /// the query is what provides the edge.
    #[test]
    fn a_held_click_reports_exactly_once() {
        let mut world = World::new();
        world.spawn((Focusable::new(1), Interaction::Pressed));

        fn probe(clicks: Query<(&Focusable, &Interaction), Changed<Interaction>>) -> Option<usize> {
            clicked_focusable(&clicks)
        }

        // Frame 1: the press edge (the component was just added ⇒ changed).
        assert_eq!(world.run_system_cached(probe).unwrap(), Some(1));
        // Frames 2..n: still held, unchanged — no re-fire.
        assert_eq!(world.run_system_cached(probe).unwrap(), None);
        assert_eq!(world.run_system_cached(probe).unwrap(), None);
    }
}
