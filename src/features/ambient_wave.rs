//! Ambient pixel wave: the Kissaten background layer.
//!
//! An animated field of discrete pixel grains whose density and weight follow
//! slow diagonal waves — warmth and identity in menus, near-subliminal life
//! during play, celebration on the results screen. Purely cosmetic: the layer
//! carries no gameplay information and the client is fully playable with it
//! disabled (the Options "Background" toggle; the dev VFX panel has a second
//! switch). When tournament-clean and reduced-motion modes land they map onto
//! the same disable path.
//!
//! How it renders: a dedicated background camera (order −100) draws one quad
//! on a reserved [`RenderLayers`] slot; every game camera composites over it
//! by not clearing (`Camera::clear_color: None`). The quad's texture is a
//! CPU-painted grain
//! lattice regenerated at a stepped ~10 Hz cadence — grains appear, disappear,
//! and reorganize rather than slide (nearest sampling keeps edges hard). The
//! board interior never shows it: the field has an opaque `BG` backplate
//! (`session::render::setup_scene`), so a pixel diff of the interior with the
//! layer on versus off is empty.
//!
//! Reactivity (sustained states only, eased over seconds — never one-shot
//! events): the surface sets the intensity budget (menus visible, in-match at
//! the edge of perception, results full); a live back-to-back chain slowly
//! tints the heaviest grains toward amber and drains after it ends; a stack
//! in the danger zone calms the wave — motion slows and density thins, it
//! never speeds up or brightens. Attack and garbage stay the meter's channel;
//! the layer never touches `ATTACK` or piece colors.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::GameState;
use crate::session::{HumanSeat, Seat, SeatSnapshot, SessionPhase};
use crate::settings::GameSettings;

/// Render layer reserved for the background camera + quad. Nothing else may
/// use it; the game cameras stay on the default layer 0.
const WAVE_RENDER_LAYER: usize = 31;

/// Camera order of the background pass — everything else draws over it.
const WAVE_CAMERA_ORDER: isize = -100;

/// Logical pixels per texel: the grain unit. Chunky on purpose (hi-bit), and
/// it keeps the repaint at a quarter of window resolution.
const TEXEL_LOGICAL_PX: f32 = 2.0;

/// Lattice stride between grain sites, in texels. Grains grow up to
/// [`MAX_GRAIN_SIDE`] texels, so sites never touch at any intensity.
const SITE_STRIDE: usize = 4;

/// Largest grain side, in texels.
const MAX_GRAIN_SIDE: usize = 3;
// Compile-time invariant: the heaviest grain never touches its neighbors.
const _: () = assert!(MAX_GRAIN_SIDE < SITE_STRIDE);

/// Texture refresh cadence. Stepped, not continuous — the house ~10 Hz.
const TICK_SECONDS: f32 = 0.1;

/// Surface intensity budgets (AW-6..AW-8): menus clearly visible, in-match at
/// the edge of perception, results full for the crescendo.
const LEVEL_MENU: f32 = 0.8;
const LEVEL_MATCH: f32 = 0.12;
const LEVEL_RESULTS: f32 = 1.0;

/// Easing time constants, seconds (state transitions smooth over seconds).
const TAU_LEVEL: f32 = 1.5;
const TAU_CALM: f32 = 0.8;
const TAU_AMBER: f32 = 2.5;

/// Wave shape. BOTH octaves travel along the same diagonal (cross-angled
/// octaves interfere into blobs, not waves); their different wavelengths and
/// speeds roll a slow beat envelope through the bands. A third, gentle
/// modulation runs ALONG the crests so bands undulate in strength without
/// losing their direction. All cycles ≥ 8 s.
const CYCLE_PRIMARY_SECONDS: f32 = 11.0;
const CYCLE_SECONDARY_SECONDS: f32 = 17.0;
const CYCLE_BREADTH_SECONDS: f32 = 29.0;
const WAVELENGTH_PRIMARY_TEXELS: f32 = 110.0;
const WAVELENGTH_SECONDARY_TEXELS: f32 = 47.0;
const WAVELENGTH_BREADTH_TEXELS: f32 = 260.0;
/// Crest sharpening: raising the band profile to this power empties the
/// troughs so crests read as distinct travelling bands, not an even wash.
const CREST_GAMMA: f32 = 1.6;

/// Grain ink: a dark warm neutral two steps under `bg` (#211E1B). Defined
/// here, not in `theme` — it is this layer's only private color.
const GRAIN_INK: [u8; 3] = [0x21, 0x1E, 0x1B];

/// Amber for the B2B tint of the heaviest grains (`theme::ACCENT` as bytes).
const GRAIN_AMBER: [u8; 3] = [0xD9, 0xA6, 0x48];

/// The 4×4 ordered Bayer matrix, the reference crosshatch dither.
const BAYER_4X4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

pub struct AmbientWavePlugin;

impl Plugin for AmbientWavePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaveState>()
            .add_systems(Startup, spawn_wave_layer)
            .add_systems(Update, (drive_wave_targets, animate_wave).chain());
    }
}

/// The layer's continuous state: eased intensities and the stepped clock.
#[derive(Resource)]
struct WaveState {
    /// Wave-phase clock in seconds. Advances with frame time, slowed by calm —
    /// distinct from `Time` so danger can still the motion without a pop.
    phase: f32,
    /// Stepped repaint clock.
    tick: Timer,
    /// Smoothed surface intensity (0..=1) and its current target.
    level: f32,
    level_target: f32,
    /// Smoothed danger calmness (0 = lively, 1 = stilled) and target.
    calm: f32,
    calm_target: f32,
    /// Smoothed B2B amber tint (0..=1) and target.
    amber: f32,
    amber_target: f32,
    /// Texel dimensions of the current texture (tracks window resizes).
    size: UVec2,
}

impl Default for WaveState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            tick: Timer::from_seconds(TICK_SECONDS, TimerMode::Repeating),
            level: 0.0,
            level_target: 0.0,
            calm: 0.0,
            calm_target: 0.0,
            amber: 0.0,
            amber_target: 0.0,
            size: UVec2::ZERO,
        }
    }
}

/// The background quad carrying the grain texture.
#[derive(Component)]
struct WaveQuad;

/// Spawn the background camera and its quad. The camera clears with the
/// global `ClearColor` (the Kissaten ground) and renders first; every game
/// camera composites over it. UI is unaffected: with no `IsDefaultUiCamera`
/// marker anywhere, Bevy targets the highest-order camera, never this one.
fn spawn_wave_layer(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        Camera2d,
        Camera {
            order: WAVE_CAMERA_ORDER,
            ..default()
        },
        RenderLayers::layer(WAVE_RENDER_LAYER),
    ));
    // A 1×1 transparent placeholder; the first animate tick sizes it to the
    // window. RenderAssetUsages keep the CPU copy so it can be repainted.
    let image = images.add(blank_image(UVec2::ONE));
    commands.spawn((
        WaveQuad,
        Sprite {
            image,
            custom_size: Some(Vec2::ONE),
            ..default()
        },
        Transform::default(),
        RenderLayers::layer(WAVE_RENDER_LAYER),
    ));
}

/// A transparent RGBA texture of `size` texels, repaintable from the CPU.
fn blank_image(size: UVec2) -> Image {
    Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
}

/// Read the sustained game state into the layer's targets. Surfaces set the
/// intensity budget; a live B2B chain arms the amber tint; a stack in the
/// danger zone arms the calm. One-shot events are deliberately never read.
fn drive_wave_targets(
    mut state: ResMut<WaveState>,
    game_state: Res<State<GameState>>,
    phase: Option<Res<State<SessionPhase>>>,
    seats: Query<(&Seat, &SeatSnapshot, Option<&HumanSeat>)>,
) {
    let in_match = *game_state.get() == GameState::Session;
    state.level_target = if in_match {
        if phase
            .as_deref()
            .is_some_and(|p| *p.get() == SessionPhase::Over)
        {
            LEVEL_RESULTS
        } else {
            LEVEL_MATCH
        }
    } else if *game_state.get() == GameState::Loading {
        0.0
    } else {
        LEVEL_MENU
    };

    // The human seat's board drives reactivity; in bot-vs-bot, any seat's
    // does (the watcher's ambience may as well follow the action).
    let (mut b2b, mut danger) = (false, false);
    if in_match {
        let mut chosen: Option<(&SeatSnapshot, bool)> = None;
        for (_, snapshot, human) in &seats {
            let is_human = human.is_some();
            if is_human || chosen.is_none_or(|(_, was_human)| !was_human) {
                chosen = Some((snapshot, is_human));
            }
        }
        if let Some((snapshot, _)) = chosen {
            b2b = snapshot.0.back_to_back_active;
            danger = crate::session::render::stack_peak_row(&snapshot.0) >= DANGER_ROW;
        }
    }
    state.amber_target = if b2b { 1.0 } else { 0.0 };
    state.calm_target = if danger { 1.0 } else { 0.0 };
}

/// The board row at which the stack counts as "in the danger zone" — the top
/// four visible rows, matching the frame-warming pass in `session::render`.
const DANGER_ROW: isize = 16;

/// Ease, advance the stepped clock, and repaint the grain texture.
fn animate_wave(
    time: Res<Time>,
    settings: Res<GameSettings>,
    toggles: Res<crate::vfx::VfxToggles>,
    mut state: ResMut<WaveState>,
    mut images: ResMut<Assets<Image>>,
    window: Single<&Window>,
    quad: Single<(&mut Sprite, &mut Visibility), With<WaveQuad>>,
) {
    let (mut sprite, mut visibility) = quad.into_inner();
    let enabled = settings.background_enabled && toggles.ambient;
    let target_visibility = if enabled {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    if *visibility != target_visibility {
        *visibility = target_visibility;
    }
    if !enabled {
        return;
    }

    // Smooth every reactive quantity each frame (transitions over seconds,
    // no pops), then advance the phase clock slowed by calm.
    let dt = time.delta_secs();
    state.level = approach(state.level, state.level_target, dt, TAU_LEVEL);
    state.calm = approach(state.calm, state.calm_target, dt, TAU_CALM);
    state.amber = approach(state.amber, state.amber_target, dt, TAU_AMBER);
    state.phase += dt * (1.0 - 0.75 * state.calm);

    // Repaint only on the stepped clock — the texture, not the transform,
    // is what moves, so motion reads as discrete reorganization.
    if !state.tick.tick(time.delta()).just_finished() {
        return;
    }

    // Track the window in texels; reallocate only on resize.
    let texels = UVec2::new(
        (window.width() / TEXEL_LOGICAL_PX).ceil().max(1.0) as u32,
        (window.height() / TEXEL_LOGICAL_PX).ceil().max(1.0) as u32,
    );
    if texels != state.size {
        state.size = texels;
        sprite.image = images.add(blank_image(texels));
        sprite.custom_size = Some(texels.as_vec2() * TEXEL_LOGICAL_PX);
    }
    let Some(image) = images.get_mut(&sprite.image) else {
        return;
    };
    let Some(buffer) = image.data.as_mut() else {
        return;
    };
    paint_grains(
        buffer,
        texels.x as usize,
        texels.y as usize,
        state.phase,
        state.level * (1.0 - 0.55 * state.calm),
        state.amber,
    );
}

/// Exponential approach of `current` toward `target` with time constant
/// `tau`: frame-rate independent, never overshoots (motion stays mechanical).
fn approach(current: f32, target: f32, dt: f32, tau: f32) -> f32 {
    current + (target - current) * (1.0 - (-dt / tau.max(f32::EPSILON)).exp())
}

/// The wave intensity field at a lattice site, in 0..=1.
///
/// `x + y` is the distance along the propagation diagonal: both octaves ride
/// it (same direction, different wavelength and speed, so a beat envelope
/// travels through the bands), and the profile is sharpened by
/// [`CREST_GAMMA`] so the space between crests goes genuinely sparse.
/// `x - y` runs along a crest; the breadth term modulates band strength
/// there by a gentle 15% so the wavefronts breathe without dissolving into
/// clumps.
fn wave_intensity(site_x: usize, site_y: usize, phase: f32, level: f32) -> f32 {
    use std::f32::consts::TAU;
    let (x, y) = (
        site_x as f32 * SITE_STRIDE as f32,
        site_y as f32 * SITE_STRIDE as f32,
    );
    let along = x + y;
    let primary = 0.5
        + 0.5 * (TAU * (along / WAVELENGTH_PRIMARY_TEXELS - phase / CYCLE_PRIMARY_SECONDS)).sin();
    let secondary = 0.5
        + 0.5
            * (TAU * (along / WAVELENGTH_SECONDARY_TEXELS - phase / CYCLE_SECONDARY_SECONDS)).sin();
    let band = (0.65 * primary + 0.35 * secondary).powf(CREST_GAMMA);
    let breadth = 0.85
        + 0.15
            * (TAU * ((x - y) / WAVELENGTH_BREADTH_TEXELS + phase / CYCLE_BREADTH_SECONDS)).sin();
    (level * band * breadth).clamp(0.0, 1.0)
}

/// Ordered-dither threshold for a lattice site, in (0, 1).
fn bayer_threshold(site_x: usize, site_y: usize) -> f32 {
    (BAYER_4X4[site_y % 4][site_x % 4] as f32 + 0.5) / 16.0
}

/// Grain side for an intensity: weight and density read as one phenomenon.
fn grain_side(intensity: f32) -> usize {
    1 + usize::from(intensity > 0.45) + usize::from(intensity > 0.75)
}

/// Repaint the whole grain field into `buffer` (RGBA, `width`×`height`
/// texels). Returns the number of grains drawn (the tests' density probe).
///
/// Grains sit on a fixed lattice ([`SITE_STRIDE`]); a site is inked when the
/// wave intensity beats its Bayer threshold, which yields the crosshatch at
/// mid intensities. The heaviest grains lerp toward amber by `amber`.
fn paint_grains(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    phase: f32,
    level: f32,
    amber: f32,
) -> usize {
    buffer.fill(0);
    if level <= f32::EPSILON {
        return 0;
    }
    let mut grains = 0;
    for site_y in 0..height.div_ceil(SITE_STRIDE) {
        for site_x in 0..width.div_ceil(SITE_STRIDE) {
            let intensity = wave_intensity(site_x, site_y, phase, level);
            if intensity <= bayer_threshold(site_x, site_y) {
                continue;
            }
            let side = grain_side(intensity);
            let ink = if side == MAX_GRAIN_SIDE && amber > 0.0 {
                lerp_rgb(GRAIN_INK, GRAIN_AMBER, 0.8 * amber)
            } else {
                GRAIN_INK
            };
            fill_square(
                buffer,
                width,
                height,
                site_x * SITE_STRIDE,
                site_y * SITE_STRIDE,
                side,
                ink,
            );
            grains += 1;
        }
    }
    grains
}

/// Component-wise sRGB lerp, `t` clamped to 0..=1.
fn lerp_rgb(from: [u8; 3], to: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|i| (from[i] as f32 + (to[i] as f32 - from[i] as f32) * t).round() as u8)
}

/// Ink an axis-aligned `side`×`side` square at texel `(x, y)`, clipped to the
/// buffer — hard edges, fully opaque.
fn fill_square(
    buffer: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    side: usize,
    ink: [u8; 3],
) {
    for ty in y..(y + side).min(height) {
        for tx in x..(x + side).min(width) {
            let at = (ty * width + tx) * 4;
            buffer[at] = ink[0];
            buffer[at + 1] = ink[1];
            buffer[at + 2] = ink[2];
            buffer[at + 3] = 0xFF;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grains_at(level: f32, amber: f32) -> (usize, Vec<u8>) {
        let (w, h) = (64, 48);
        let mut buffer = vec![0u8; w * h * 4];
        let count = paint_grains(&mut buffer, w, h, 3.7, level, amber);
        (count, buffer)
    }

    #[test]
    fn density_grows_with_intensity() {
        let sparse = grains_at(0.12, 0.0).0;
        let mid = grains_at(0.5, 0.0).0;
        let dense = grains_at(1.0, 0.0).0;
        assert!(sparse < mid && mid < dense, "{sparse} < {mid} < {dense}");
        // The in-match cap reads as scattered isolated grains, not a texture.
        let sites = (64usize / 4) * (48 / 4);
        assert!(sparse * 5 < sites, "in-match density must stay subliminal");
    }

    #[test]
    fn zero_level_paints_nothing_and_clears_the_buffer() {
        let (w, h) = (16, 16);
        let mut buffer = vec![0xAAu8; w * h * 4];
        assert_eq!(paint_grains(&mut buffer, w, h, 1.0, 0.0, 0.0), 0);
        assert!(buffer.iter().all(|&b| b == 0), "stale grains must clear");
    }

    #[test]
    fn grain_side_is_monotonic_and_bounded() {
        let sides: Vec<usize> = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0]
            .iter()
            .map(|&i| grain_side(i))
            .collect();
        assert!(sides.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(sides.first(), Some(&1));
        assert_eq!(sides.last(), Some(&MAX_GRAIN_SIDE));
    }

    #[test]
    fn calm_thins_the_field() {
        // The animate system scales level by (1 - 0.55*calm); fully calm must
        // visibly thin the same surface budget.
        let lively = grains_at(0.8, 0.0).0;
        let calmed = grains_at(0.8 * (1.0 - 0.55), 0.0).0;
        assert!(calmed < lively, "{calmed} < {lively}");
    }

    #[test]
    fn the_layer_is_two_color_at_rest_and_amber_tints_only_heavy_grains() {
        let (_, buffer) = grains_at(1.0, 0.0);
        for texel in buffer.chunks_exact(4) {
            assert!(
                texel == [0, 0, 0, 0] || texel[..3] == GRAIN_INK,
                "at rest the layer is ink-on-ground only, got {texel:?}"
            );
        }
        let (_, tinted) = grains_at(1.0, 1.0);
        let expected = lerp_rgb(GRAIN_INK, GRAIN_AMBER, 0.8);
        assert!(
            tinted.chunks_exact(4).any(|t| t[..3] == expected),
            "a sustained B2B chain must tint the heaviest grains"
        );
        // Never attack red, never anything brighter than the accent lerp.
        assert!(
            tinted
                .chunks_exact(4)
                .all(|t| t == [0, 0, 0, 0] || t[..3] == GRAIN_INK || t[..3] == expected),
        );
    }

    #[test]
    fn the_field_is_a_directional_wave_not_blobs() {
        // Spread sampled ACROSS the bands (along the x+y propagation
        // diagonal) must dwarf the spread ALONG a single crest (x - y
        // varying, x + y fixed) — that anisotropy is what makes the layer
        // read as travelling waves instead of clumps.
        let spread = |samples: &[f32]| {
            samples.iter().fold(f32::MIN, |a, &b| a.max(b))
                - samples.iter().fold(f32::MAX, |a, &b| a.min(b))
        };
        let across: Vec<f32> = (0..60).map(|i| wave_intensity(i, i, 3.7, 1.0)).collect();
        let along: Vec<f32> = (0..60)
            .map(|i| wave_intensity(i, 60 - i, 3.7, 1.0))
            .collect();
        assert!(
            spread(&across) > 3.0 * spread(&along),
            "across-band spread {} must dwarf along-crest spread {}",
            spread(&across),
            spread(&along)
        );
    }

    #[test]
    fn bayer_thresholds_are_distinct_and_interior() {
        let mut seen = std::collections::BTreeSet::new();
        for y in 0..4 {
            for x in 0..4 {
                let t = bayer_threshold(x, y);
                assert!(t > 0.0 && t < 1.0);
                seen.insert((t * 16.0) as u32);
            }
        }
        assert_eq!(seen.len(), 16, "all 16 dither levels must be distinct");
    }

    #[test]
    fn approach_converges_without_overshoot() {
        let mut value: f32 = 0.0;
        for _ in 0..600 {
            value = approach(value, 1.0, 1.0 / 60.0, TAU_LEVEL);
            assert!((0.0..=1.0).contains(&value));
        }
        assert!(value > 0.99, "ten seconds must converge, got {value}");
        // A single huge step still lands inside the range.
        assert!(approach(0.0, 1.0, 100.0, TAU_LEVEL) <= 1.0);
    }

    #[test]
    fn fill_square_clips_at_the_buffer_edge() {
        let (w, h) = (8, 8);
        let mut buffer = vec![0u8; w * h * 4];
        // A 3×3 grain at the far corner must clip, not panic or wrap.
        fill_square(&mut buffer, w, h, 7, 7, 3, GRAIN_INK);
        let inked = buffer.chunks_exact(4).filter(|t| t[3] == 0xFF).count();
        assert_eq!(inked, 1, "only the in-bounds texel is inked");
    }
}
