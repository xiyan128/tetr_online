// CRT / retro-arcade post-process for tetr_online.
//
// Runs after tonemapping on the fully-composited 2D frame and stacks the classic
// CRT cues: barrel curvature (with a black bezel beyond the glass), radial
// chromatic aberration, scanlines + a subtle aperture-grille mask, a corner
// vignette, and a faint mains-hum flicker. Everything is driven by a single
// `CrtSettings` uniform so the look is tunable from the main world at runtime.
//
// The uniform is two vec4s — 16-byte aligned by construction, so it is valid
// under WebGL2's stricter std140 layout with no conditional padding.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct CrtSettings {
    // x: time (seconds), y: curvature, z: scanline intensity, w: aberration
    params_a: vec4<f32>,
    // x: vignette, y: mask intensity, z: brightness, w: (unused)
    params_b: vec4<f32>,
}
@group(0) @binding(2) var<uniform> settings: CrtSettings;

const PI: f32 = 3.14159265;
// Fixed scanline count, so the lines stay visible and stable regardless of the
// window resolution (tying them to the pixel height would alias into moiré).
const SCANLINES: f32 = 240.0;
// Fixed aperture-grille column count. Resolution-independent (keyed to uv, not the
// framebuffer pixel grid) so the phosphor triads never moiré against the display.
const MASK_COLUMNS: f32 = 180.0;

// Barrel-distort UVs around the screen center to fake the bulge of CRT glass.
// `amount` 0 == perfectly flat. No overscan zoom: the scale stays 1:1 (so pointer
// hit-testing still lines up with the UI), and the slight corner bulge is handled
// by the clamp-to-edge sampler, which fills it with the dark screen background
// rather than a black border.
fn curve_uv(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    var c = uv * 2.0 - 1.0;            // recenter to [-1, 1]
    let offset = c.yx * c.yx * amount; // push each axis out by the other's square
    c = c + c * offset;
    return c * 0.5 + 0.5;              // back to [0, 1]
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let time = settings.params_a.x;
    let curvature = settings.params_a.y;
    let scanline_intensity = settings.params_a.z;
    let aberration = settings.params_a.w;
    let vignette_amt = settings.params_b.x;
    let mask_intensity = settings.params_b.y;
    let brightness = settings.params_b.z;
    let enabled = settings.params_b.w;

    // Effect toggled off (dev panel): pass the frame straight through.
    if (enabled < 0.5) {
        return textureSample(screen_texture, texture_sampler, in.uv);
    }

    let uv = curve_uv(in.uv, curvature);

    // No hard bezel: the clamp-to-edge sampler resolves the slight corner bulge to
    // the dark screen background (not black), and the vignette below softens it.

    // Chromatic aberration: pull the red and blue channels apart radially so the
    // fringing grows toward the edges, like a misconverged electron gun.
    let to_center = uv - vec2<f32>(0.5);
    let shift = to_center * aberration;
    var color: vec3<f32>;
    color.r = textureSample(screen_texture, texture_sampler, uv + shift).r;
    color.g = textureSample(screen_texture, texture_sampler, uv).g;
    color.b = textureSample(screen_texture, texture_sampler, uv - shift).b;

    // Scanlines: a soft dark band every other line, rolling very slowly upward.
    let scan = 0.5 + 0.5 * sin((uv.y * SCANLINES + time * 0.5) * PI);
    color = color * (1.0 - scanline_intensity * (1.0 - scan));

    // Aperture-grille mask: tint successive columns R / G / B so the image reads as
    // phosphor triads. Keyed to the curved `uv` (so the triads follow the glass) at
    // a fixed column count, which keeps it resolution-independent and moiré-free.
    let col = u32(uv.x * MASK_COLUMNS) % 3u;
    var mask = vec3<f32>(1.0 - mask_intensity);
    if (col == 0u) {
        mask.r = 1.0;
    } else if (col == 1u) {
        mask.g = 1.0;
    } else {
        mask.b = 1.0;
    }
    color = color * mask;

    // Vignette: ease the corners down. `to_center` spans ±0.5, so scale to ~0..1.
    let d = length(to_center) * 1.41421;
    color = color * clamp(1.0 - vignette_amt * d * d, 0.0, 1.0);

    // Faint flicker, and overall brightness lift to offset the darkening above.
    let flicker = 1.0 + 0.015 * sin(time * 110.0);
    color = color * brightness * flicker;

    return vec4<f32>(color, 1.0);
}
