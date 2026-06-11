//! Neon bloom / HDR glow on the gameplay camera (the Tetris-Effect look).
//!
//! WebGPU-first: Bevy 0.18's built-in bloom is not WebGL2-compatible, so this
//! whole module compiles only under the `bloom` cargo feature — present on native
//! and the WebGPU web bundle, absent from the WebGL2 bundle (see `Cargo.toml`).
//!
//! The glow itself comes from over-bright (`> 1.0`) mino colors — gated on the same
//! feature in [`crate::level::common`] — rising past the bloom threshold. Everything
//! else stays at LDR (`<= 1.0`), so the board background and UI text never bloom and
//! the image stays crisp.
//!
//! Bloom is attached only to the [`GameplayCamera`] (HDR is auto-required by the
//! `Bloom` component), so the menus keep their flat look.

use bevy::core_pipeline::tonemapping::DebandDither;
use bevy::post_process::bloom::{Bloom, BloomPrefilter};
use bevy::prelude::*;

use crate::level::common::GameplayCamera;
use crate::vfx::VfxToggles;

/// Bloom strength — enough to halo the bright minos and clear sparks without
/// drowning the board. Tunable (best dialed in against a running build).
const BLOOM_INTENSITY: f32 = 0.28;
/// Only HDR colors brighter than this bloom. At `1.0`, ordinary LDR content (the
/// dark board, UI text) never glows — only the deliberately over-bright minos and
/// sparks do.
const BLOOM_THRESHOLD: f32 = 1.0;

/// Neon bloom on the gameplay camera. Compiled only with the `bloom` feature.
pub struct BloomPlugin;

impl Plugin for BloomPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (attach_bloom, sync_bloom_toggle));
    }
}

/// Drive the bloom strength from the dev toggle: full intensity when on, zero when
/// off (a zero-intensity pass leaves the HDR view crisp without churning the
/// camera's component set). The change guard keeps it from re-triggering change
/// detection every frame.
fn sync_bloom_toggle(toggles: Res<VfxToggles>, mut cameras: Query<&mut Bloom>) {
    let target = if toggles.bloom { BLOOM_INTENSITY } else { 0.0 };
    for mut bloom in &mut cameras {
        if bloom.intensity != target {
            bloom.intensity = target;
        }
    }
}

/// Give the gameplay camera a bloom pass — and, via `Bloom`'s `#[require(Hdr)]`,
/// an HDR target — the first time it appears. The `Without<Bloom>` filter makes
/// this run exactly once per camera (and re-arm after a restart spawns a new one).
fn attach_bloom(
    mut commands: Commands,
    cameras: Query<Entity, (With<GameplayCamera>, Without<Bloom>)>,
) {
    for entity in &cameras {
        commands.entity(entity).insert((
            Bloom {
                intensity: BLOOM_INTENSITY,
                prefilter: BloomPrefilter {
                    threshold: BLOOM_THRESHOLD,
                    threshold_softness: 0.4,
                },
                ..Bloom::NATURAL
            },
            // Smooth the glow gradients so the halo doesn't band on 8-bit output.
            DebandDither::Enabled,
        ));
    }
}
