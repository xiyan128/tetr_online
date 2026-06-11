//! Feature plugins: one self-contained plugin per gameplay/UI feature.
//!
//! Each feature lives in its OWN file with its own Bevy plugin, all wired into the
//! app via [`FeaturesPlugin`]. Keeping them separate means adding or changing one
//! feature stays local to its file — no central registration churn.
//!
//! | feature        | file                          | plugin                     | fills |
//! |----------------|-------------------------------|----------------------------|-------|
//! | ambient_wave   | `features/ambient_wave.rs`    | `AmbientWavePlugin`        | the Kissaten pixel-grain background layer |
//! | options        | `features/options.rs`         | `OptionsPlugin`            | Options-screen widgets that mutate `GameSettings` |
//! | help           | `features/help.rs`            | `HelpPlugin`               | Help-screen controls/about content |
//! | notifications  | `features/notifications.rs`   | `NotificationsPlugin`      | line-clear flash + hard-drop trail effects |
//! | screen_shake   | `features/screen_shake.rs`    | `ScreenShakePlugin`        | trauma-based camera shake on impacts |
//! | hit_stop       | `features/hit_stop.rs`        | `HitStopPlugin`            | brief world freeze on Tetris / T-spin |
//! | sfx            | `features/sfx.rs`             | `SfxPlugin`                | music + volume (reads `GameSettings`) |
//! | high_scores    | `features/high_scores.rs`     | `HighScoresFeaturePlugin`  | record runs + render leaderboard tables |

use bevy::prelude::*;

mod ambient_wave;
mod help;
pub(crate) mod high_scores;
pub(crate) mod hit_stop;
mod notifications;
pub(crate) mod options;
pub(crate) mod screen_shake;
mod sfx;

/// Registers every feature plugin.
pub struct FeaturesPlugin;

impl Plugin for FeaturesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ambient_wave::AmbientWavePlugin,
            options::OptionsPlugin,
            help::HelpPlugin,
            notifications::NotificationsPlugin,
            screen_shake::ScreenShakePlugin,
            hit_stop::HitStopPlugin,
            sfx::SfxPlugin,
            high_scores::HighScoresFeaturePlugin,
        ));
    }
}
