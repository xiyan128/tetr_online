//! Feature plugin stubs (M1 fan-out seam).
//!
//! Each feature lives in its OWN file with an empty-but-registered Bevy plugin
//! that a fan-out agent fleshes out without touching the others. They are wired
//! into the app now (via [`FeaturesPlugin`]) so adding behavior is purely
//! additive — no central registration churn.
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
mod options;
mod pause;
mod sfx;

/// Registers every feature stub plugin.
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
