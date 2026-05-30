//! Keyboard focus-navigation helper for menu screens (M1 shared UI).
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
    mut buttons: Query<(&Focusable, &Interaction, &mut BackgroundColor)>,
) {
    // Move the cursor to the pointed row — but weigh a press and a hover
    // differently. A press is an explicit action, so it always claims the cursor;
    // a hover only does on a frame the pointer moved (see the doc comment).
    let pointer_moved = pointer_motion.delta != Vec2::ZERO;
    if let Some(index) = buttons.iter().find_map(|(focusable, interaction, _)| {
        let claims_cursor = match *interaction {
            Interaction::Pressed => true,
            Interaction::Hovered => pointer_moved,
            Interaction::None => false,
        };
        claims_cursor.then_some(focusable.index)
    }) {
        list.index = index;
    }

    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        list.move_by(1);
    }
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        list.move_by(-1);
    }

    for (focusable, interaction, mut color) in &mut buttons {
        *color = if *interaction == Interaction::Pressed {
            theme::BUTTON_PRESSED.into()
        } else if focusable.index == list.index {
            theme::BUTTON_FOCUSED.into()
        } else {
            theme::BUTTON_NORMAL.into()
        };
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

/// The 0-based index of a focusable menu button being clicked (pressed) this
/// frame, if any — the mouse counterpart to [`read_nav_action`]'s `Select`.
///
/// A screen's activation handler treats this exactly like a keyboard Select:
/// `read_nav_action(..).or_else(|| clicked_focusable(&clicks).map(NavAction::Select))`.
/// [`focus_navigation`] already moves the cursor to the hovered row, so the click
/// and the highlight always agree.
pub fn clicked_focusable(buttons: &Query<(&Focusable, &Interaction)>) -> Option<usize> {
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
}
