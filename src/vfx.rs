//! Runtime on/off switches for the visual-effects stack, with a dev-only egui
//! panel to flip them live.
//!
//! [`VfxToggles`] gates each effect at runtime. In a normal build it is simply
//! all-on and never changes (the run conditions and sync systems read it but
//! nothing writes it). With `--features dev` a small "Visual FX" egui window flips
//! the flags while the game runs, so each effect can be A/B'd in isolation.
//!
//! The gameplay-juice effects (shake, hit-stop) are gated with run conditions; the
//! bloom render effect instead reads the resource directly to drive a
//! zero-intensity path, since its camera component can't simply be switched off
//! mid-frame.

use bevy::prelude::*;

/// Per-effect on/off switches. All on by default.
#[derive(Resource, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub(crate) struct VfxToggles {
    pub shake: bool,
    pub hit_stop: bool,
    pub bloom: bool,
    /// The ambient pixel-wave background (`features::ambient_wave`). ANDed
    /// with the player-facing `GameSettings::background_enabled`.
    pub ambient: bool,
}

impl Default for VfxToggles {
    fn default() -> Self {
        Self {
            shake: true,
            hit_stop: true,
            bloom: true,
            ambient: true,
        }
    }
}

// Run conditions for the juice effects. Defined here so each feature plugin just
// references one shared predicate.
pub(crate) fn shake_enabled(toggles: Res<VfxToggles>) -> bool {
    toggles.shake
}
pub(crate) fn hit_stop_enabled(toggles: Res<VfxToggles>) -> bool {
    toggles.hit_stop
}

/// Dev-only "Visual FX" panel: a checkbox per effect, wired straight to
/// [`VfxToggles`]. Drawn alongside the world inspector in `EguiPrimaryContextPass`.
#[cfg(feature = "dev")]
pub(crate) fn vfx_debug_panel(
    mut egui_ctx: Query<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>,
    mut toggles: ResMut<VfxToggles>,
) {
    use bevy_egui::egui;
    let Ok(mut ctx) = egui_ctx.single_mut() else {
        return;
    };
    egui::Window::new("Visual FX")
        .default_open(true)
        .show(ctx.get_mut(), |ui| {
            ui.label("Toggle each effect live:");
            ui.checkbox(&mut toggles.shake, "Screen shake");
            ui.checkbox(&mut toggles.hit_stop, "Hit-stop (Tetris / T-spin)");
            ui.checkbox(&mut toggles.ambient, "Ambient background");
            #[cfg(feature = "bloom")]
            ui.checkbox(&mut toggles.bloom, "Neon bloom");
            #[cfg(not(feature = "bloom"))]
            {
                let mut off = false;
                ui.add_enabled(
                    false,
                    egui::Checkbox::new(&mut off, "Neon bloom (`--features bloom` builds only)"),
                );
            }
        });
}
