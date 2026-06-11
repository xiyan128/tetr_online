//! Camera / render-pipeline visual effects, as opposed to the gameplay-driven
//! juice in [`crate::features`].
//!
//! The Kissaten core skin is flat by rule — no scanlines, curvature, or glow;
//! heritage is expressed through palette, type, and motion. The old CRT pass
//! is gone for good (it was costume). What remains:
//!
//! * [`bloom`] — neon glow on the gameplay camera via an HDR pass. An optional
//!   skin, never the core look: behind the `bloom` cargo feature and absent
//!   from every default build.
//!
//! [`PostFxPlugin`] wires up whichever passes the current build supports.

use bevy::prelude::*;

#[cfg(feature = "bloom")]
pub(crate) mod bloom;

/// Registers the render-pipeline visual effects.
pub struct PostFxPlugin;

impl Plugin for PostFxPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "bloom")]
        app.add_plugins(bloom::BloomPlugin);
        #[cfg(not(feature = "bloom"))]
        let _ = app;
    }
}
