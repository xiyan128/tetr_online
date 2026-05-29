//! Menu screen shells (A1.1).
//!
//! One plugin per non-gameplay screen — Title, MainMenu, ModeSelect, Options,
//! Help, HighScores. Each spawns a keyboard-navigable placeholder on enter and
//! tears it down on exit (`DespawnOnExit`). Navigation actually works (Up/Down
//! move focus, Enter selects, Esc backs out); the *content* of Options/Help/
//! HighScores is intentionally a stub that the corresponding feature plugin in
//! [`crate::features`] fleshes out into its own file.
//!
//! [`crate::features`]: crate::features

use bevy::prelude::*;

mod help;
mod high_scores;
mod main_menu;
mod mode_select;
mod options;
mod title;

// Screen-root markers, re-exported so the corresponding feature plugins in
// `crate::features` can attach their content under the right entity.
pub(crate) use help::HelpRoot;
pub(crate) use high_scores::HighScoresRoot;
pub(crate) use options::OptionsRoot;

/// Registers every menu screen plugin.
pub struct ScreensPlugin;

impl Plugin for ScreensPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            title::TitleScreenPlugin,
            main_menu::MainMenuPlugin,
            mode_select::ModeSelectPlugin,
            options::OptionsScreenPlugin,
            help::HelpScreenPlugin,
            high_scores::HighScoresScreenPlugin,
        ));
    }
}
