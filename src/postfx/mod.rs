//! Camera / render-pipeline visual effects, as opposed to the gameplay-driven
//! juice in [`crate::features`].
//!
//! These plugins reach below the sprite layer into Bevy's render graph and camera
//! configuration:
//!
//! * [`crt`] — a fullscreen CRT post-process pass (custom render-graph node +
//!   WGSL), applied to every 2D camera. Runs on both web backends.
//! * [`bloom`] — neon glow on the gameplay camera via an HDR pass. WebGPU/native
//!   only (Bevy's built-in bloom is not WebGL2-compatible), so it is gated behind
//!   the `bloom` cargo feature and simply absent from the WebGL2 bundle.
//!
//! [`PostFxPlugin`] wires up whichever passes the current build supports.

use bevy::prelude::*;

#[cfg(feature = "bloom")]
pub(crate) mod bloom;
pub(crate) mod crt;

/// Registers the render-pipeline visual effects.
pub struct PostFxPlugin;

impl Plugin for PostFxPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(crt::CrtPlugin);
        #[cfg(feature = "bloom")]
        app.add_plugins(bloom::BloomPlugin);
    }
}
