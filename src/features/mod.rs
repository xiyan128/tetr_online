//! Feature plugins (M1): one self-contained plugin per gameplay/UI feature.
//!
//! Each feature lives in its OWN file with its own Bevy plugin, all wired into the
//! app via [`FeaturesPlugin`]. Keeping them separate means adding or changing one
//! feature stays local to its file — no central registration churn.
//!
//! | feature        | file                          | plugin                     | fills |
//! |----------------|-------------------------------|----------------------------|-------|
//! | pause          | `features/pause.rs`           | `PausePlugin`              | Paused overlay + Esc toggle |
//! | info_panel     | `features/info_panel.rs`      | `InfoPanelPlugin`          | in-game variant/goal/time/score panel |
//! | options        | `features/options.rs`         | `OptionsPlugin`            | Options-screen widgets that mutate `GameSettings` |
//! | help           | `features/help.rs`            | `HelpPlugin`               | Help-screen controls/about content |
//! | notifications  | `features/notifications.rs`   | `NotificationsPlugin`      | transient on-screen messages |
//! | sfx            | `features/sfx.rs`             | `SfxPlugin`                | music + volume (reads `GameSettings`) |
//! | high_scores    | `features/high_scores.rs`     | `HighScoresFeaturePlugin`  | record runs + render leaderboard tables |

use bevy::prelude::*;

mod help;
mod high_scores;
mod info_panel;
mod notifications;
pub(crate) mod options;
mod pause;
mod sfx;

/// Registers every feature plugin.
pub struct FeaturesPlugin;

impl Plugin for FeaturesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            pause::PausePlugin,
            info_panel::InfoPanelPlugin,
            options::OptionsPlugin,
            help::HelpPlugin,
            notifications::NotificationsPlugin,
            sfx::SfxPlugin,
            high_scores::HighScoresFeaturePlugin,
        ));
    }
}
