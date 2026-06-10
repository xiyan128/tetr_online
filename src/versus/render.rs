//! Versus rendering: two board groups, parented per seat.
//!
//! Placeholder registration — the reconcilers land with the rendering pass of
//! the versus strike (see `docs/adr-versus-mode-ui.md`, Decision 5).

use bevy::prelude::*;

pub struct VersusRenderPlugin;

impl Plugin for VersusRenderPlugin {
    fn build(&self, _app: &mut App) {}
}
