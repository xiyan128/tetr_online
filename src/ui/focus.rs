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

/// Move the focus cursor with Up/Down (and W/S) and restyle focusable buttons so
/// the focused row reads as highlighted. Generic over a screen marker `M` so each
/// screen's lists are isolated.
///
/// `M` is the screen-root marker component; the [`FocusList`] is expected on the
/// same entity. Call `app.add_systems(Update, focus_navigation::<MyScreen>.run_if(in_state(...)))`.
///
/// `Single` makes the scheduler skip this entirely unless exactly one `FocusList`
/// with marker `M` exists — which is also the only state in which the focusable
/// rows (and thus the restyle below) are meaningful, so skipping is correct.
pub fn focus_navigation<M: Component>(
    keys: Res<ButtonInput<KeyCode>>,
    mut list: Single<&mut FocusList, With<M>>,
    mut buttons: Query<(&Focusable, &mut BackgroundColor)>,
) {
    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        list.move_by(1);
    }
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        list.move_by(-1);
    }

    for (focusable, mut color) in &mut buttons {
        *color = if focusable.index == list.index {
            theme::BUTTON_FOCUSED.into()
        } else {
            theme::BUTTON_NORMAL.into()
        };
    }
}

/// Detect Enter/Space (select focused) and Esc (back), returning a [`NavAction`]
/// for the caller to handle against the current [`FocusList`].
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
