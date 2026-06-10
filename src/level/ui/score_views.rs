//! On-playfield score readouts: score, line count, and last clear type.
//!
//! Spawns the text labels and updates them from the [`Scorer`] resource each
//! frame while in-game.

use crate::assets::GameAssets;
use crate::level::common::{to_translation, LevelConfig};
use crate::level::score::{ScoreType, ScoreTypes, Scorer};
use crate::level::ui::calc_ui_offset;
use crate::GameState;
use bevy::color::Alpha;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ScoreText;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct LineCountText;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct ScoreTypeText;

fn make_line_count_text(scorer: &Scorer) -> String {
    format!("LINES: {}", scorer.lines)
}

fn make_score_text(scorer: &Scorer) -> String {
    format!("SCORE: {}", scorer.score)
}

/// Spawn one on-board text readout at board `row`, tagged with `marker` and
/// despawned on exit from Playing — the shared body behind [`spawn_score_text`]
/// and [`spawn_line_count_text`].
fn spawn_readout(
    commands: &mut Commands,
    config: &LevelConfig,
    game_assets: &GameAssets,
    row: isize,
    marker: impl Component,
    text: String,
) {
    let offset = Vec3::new(calc_ui_offset(config), 0., 0.);

    commands
        .spawn((
            Text2d::new(text),
            TextFont {
                font: game_assets.font.clone(),
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_translation(
                to_translation(config.board_width as isize, row, config.block_size) + offset,
            ),
            Anchor::TOP_LEFT,
        ))
        .insert(marker)
        .insert(DespawnOnExit(GameState::Playing));
}

pub fn spawn_score_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    spawn_readout(
        &mut commands,
        &config,
        &game_assets,
        1,
        ScoreText,
        make_score_text(&Scorer::default()),
    );
}

pub fn spawn_line_count_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    spawn_readout(
        &mut commands,
        &config,
        &game_assets,
        2,
        LineCountText,
        make_line_count_text(&Scorer::default()),
    );
}

pub fn spawn_score_type_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    let offset = -Vec3::new(calc_ui_offset(&config), 0., 0.);
    let base = to_translation(
        0,
        ((1 + config.board_height) >> 1) as isize,
        config.block_size,
    ) + offset;

    commands
        .spawn((
            Text2d::new(""),
            TextFont {
                font: game_assets.font.clone(),
                font_size: CALLOUT_BASE_FONT,
                ..default()
            },
            // Starts invisible; a clear fades it in via `animate_callout`.
            TextColor(Color::WHITE.with_alpha(0.0)),
            // Right-justified so the stacked callouts sit flush against the
            // board's left edge (the TOP_RIGHT anchor pins that edge).
            TextLayout::new_with_justify(Justify::Right),
            Transform::from_translation(base),
            Anchor::TOP_RIGHT,
            // Start expired so nothing shows until the first clear.
            Callout {
                elapsed: CALLOUT_MAX_TTL,
                base,
                style: CalloutStyle::idle(),
            },
        ))
        .insert(ScoreTypeText)
        .insert(DespawnOnExit(GameState::Playing));
}

pub fn update_score_text(mut text: Single<&mut Text2d, With<ScoreText>>, scorer: Res<Scorer>) {
    text.0 = make_score_text(&scorer);
}

pub fn update_line_count_text(
    mut text: Single<&mut Text2d, With<LineCountText>>,
    scorer: Res<Scorer>,
) {
    text.0 = make_line_count_text(&scorer);
}

// ---------------------------------------------------------------------------
// Escalating clear callout ("juice")
// ---------------------------------------------------------------------------
//
// One on-board readout, but the *bigger* the win the bigger the show: a lone
// Single barely whispers, while a Tetris, a T-Spin, a Back-to-Back, or a deep
// combo scales the font up, pops harder, shakes more, runs a hotter colour, and
// lingers longer. Every effect scales off a single "excitement" value so the
// escalation stays coherent.

/// Font size of the calmest callout (a lone Single); the biggest wins scale up
/// toward [`CALLOUT_MAX_FONT`].
const CALLOUT_BASE_FONT: f32 = 15.0;
const CALLOUT_MAX_FONT: f32 = 26.0;
/// Lifetime of the calmest callout; big wins linger toward [`CALLOUT_MAX_TTL`].
const CALLOUT_BASE_TTL: f32 = 0.85;
const CALLOUT_MAX_TTL: f32 = 1.9;
/// Fraction of a callout's life held fully opaque before it begins to fade.
const CALLOUT_HOLD_FRACTION: f32 = 0.45;
/// How long the appear-pop and the shake take to settle (seconds).
const CALLOUT_POP_SECONDS: f32 = 0.22;
const CALLOUT_SHAKE_SECONDS: f32 = 0.3;
/// Excitement at which all effects saturate (see [`excitement`]).
const CALLOUT_MAX_EXCITEMENT: f32 = 9.0;

/// Tiered presentation for one clear callout, derived in [`callout_style`].
#[derive(Clone, Copy)]
struct CalloutStyle {
    /// Resting colour (alpha is driven by the fade).
    color: Color,
    /// Font size for this callout (set once; the pop uses transform scale).
    font_size: f32,
    /// Extra scale at the instant it appears — a "stamp" that settles back to 1.
    pop: f32,
    /// Peak shake amplitude in pixels, decaying over the first slice of life.
    shake: f32,
    /// Total lifetime: held readable, then faded out.
    ttl: f32,
}

impl CalloutStyle {
    /// The resting style before any clear — neutral and fully faded.
    fn idle() -> Self {
        Self {
            color: Color::WHITE,
            font_size: CALLOUT_BASE_FONT,
            pop: 1.0,
            shake: 0.0,
            ttl: CALLOUT_BASE_TTL,
        }
    }
}

/// Per-callout animation state on the [`ScoreTypeText`] entity: time since the
/// current callout fired, its resting position, and its tiered [`CalloutStyle`].
/// Internal renderer juice — intentionally not reflected. `pub(crate)` only so it
/// can appear in the (effectively crate-internal) animator system signatures.
#[derive(Component)]
pub(crate) struct Callout {
    elapsed: f32,
    base: Vec3,
    style: CalloutStyle,
}

/// How "exciting" a clear is, summed over its labels — the single knob the size,
/// pop, shake, and linger all scale from. Tuned so a lone Single sits at the
/// floor and a Tetris, a T-Spin, a Back-to-Back, or a deep combo each push up.
fn excitement(types: &[ScoreType]) -> f32 {
    types
        .iter()
        .map(|t| match t {
            ScoreType::Single => 1.0,
            ScoreType::MiniTSpin => 2.5,
            ScoreType::Double => 2.5,
            ScoreType::Triple => 4.0,
            ScoreType::TSpin => 5.0,
            ScoreType::Tetris => 6.0,
            ScoreType::BackToBack => 2.0,
            // A combo escalates hard the longer it runs.
            ScoreType::Combo(n) => 1.0 + *n as f32,
        })
        .sum()
}

/// The callout's colour. A deliberately small palette: only the two signature
/// clears get a hue (T-Spin purple, Tetris cyan); every ordinary line clear stays
/// neutral white and lets size + shake carry the escalation instead.
fn dominant_color(types: &[ScoreType]) -> Color {
    let any = |pred: fn(&ScoreType) -> bool| types.iter().any(pred);
    if any(|t| matches!(t, ScoreType::TSpin | ScoreType::MiniTSpin)) {
        Color::srgb(0.80, 0.47, 0.97) // vivid purple — the T-Spin signature
    } else if any(|t| matches!(t, ScoreType::Tetris)) {
        Color::srgb(0.36, 0.84, 1.0) // bright cyan — the Tetris / I-piece colour
    } else {
        Color::srgb(0.94, 0.94, 0.94) // every other line clear — near-white
    }
}

/// Map a clear's labels to its tiered [`CalloutStyle`]. Everything escalates
/// along the normalised excitement `t`.
fn callout_style(types: &[ScoreType]) -> CalloutStyle {
    let t = ((excitement(types) - 1.0) / (CALLOUT_MAX_EXCITEMENT - 1.0)).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * t;
    CalloutStyle {
        color: dominant_color(types),
        font_size: lerp(CALLOUT_BASE_FONT, CALLOUT_MAX_FONT),
        pop: lerp(1.12, 1.6),
        shake: lerp(0.0, 7.0),
        ttl: lerp(CALLOUT_BASE_TTL, CALLOUT_MAX_TTL),
    }
}

/// Rewrite the callout text and (re)trigger its animation whenever a new clear is
/// scored: install the tiered [`callout_style`] and reset the clock.
/// [`animate_callout`] does the per-frame motion.
pub fn update_score_type_text(
    callout: Single<(&mut Text2d, &mut TextFont, &mut Callout), With<ScoreTypeText>>,
    mut ev_score_type: MessageReader<ScoreTypes>,
) {
    // Only the most recent clear this frame is shown (one lock per frame in
    // practice); reading drains the rest so they don't re-fire next frame.
    let Some(latest) = ev_score_type.read().last() else {
        return;
    };
    let (mut text, mut font, mut callout) = callout.into_inner();
    text.0 = latest
        .0
        .iter()
        .map(ScoreType::label)
        .collect::<Vec<_>>()
        .join("\n");
    let style = callout_style(&latest.0);
    font.font_size = style.font_size;
    callout.style = style;
    callout.elapsed = 0.0;
}

/// Animate the live callout: a stamp-pop that settles, a quick decaying shake,
/// the tier colour, and a hold-then-fade. Once past its lifetime it rests
/// invisible and centred.
pub fn animate_callout(
    callout: Single<(&mut Transform, &mut TextColor, &mut Callout), With<ScoreTypeText>>,
    time: Res<Time>,
) {
    let (mut transform, mut color, mut callout) = callout.into_inner();
    callout.elapsed += time.delta_secs();
    let CalloutStyle {
        color: base_color,
        pop,
        shake,
        ttl,
        ..
    } = callout.style;
    let elapsed = callout.elapsed;

    // Hold fully opaque, then fade to nothing over the rest of the lifetime.
    let life = (elapsed / ttl).clamp(0.0, 1.0);
    let alpha = if life < CALLOUT_HOLD_FRACTION {
        1.0
    } else {
        let fade = (life - CALLOUT_HOLD_FRACTION) / (1.0 - CALLOUT_HOLD_FRACTION);
        (1.0 - fade).clamp(0.0, 1.0)
    };
    color.0 = base_color.with_alpha(alpha);

    // Appear-pop: starts enlarged by `pop`, eases back to 1 over POP_SECONDS.
    let pop_t = (elapsed / CALLOUT_POP_SECONDS).clamp(0.0, 1.0);
    let scale = 1.0 + (pop - 1.0) * (1.0 - pop_t) * (1.0 - pop_t);
    transform.scale = Vec3::splat(scale);

    // Shake: high-frequency jitter (mixed sines — no RNG needed) that decays away.
    let shake_decay = (1.0 - elapsed / CALLOUT_SHAKE_SECONDS).clamp(0.0, 1.0);
    let amp = shake * shake_decay;
    let ox = (elapsed * 91.0).sin() * 0.7 + (elapsed * 47.0).sin() * 0.3;
    let oy = (elapsed * 113.0).cos() * 0.7 + (elapsed * 67.0).cos() * 0.3;
    transform.translation = callout.base + Vec3::new(ox * amp, oy * amp, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excitement_grows_with_clear_size() {
        let e = excitement;
        assert!(e(&[ScoreType::Single]) < e(&[ScoreType::Double]));
        assert!(e(&[ScoreType::Double]) < e(&[ScoreType::Triple]));
        assert!(e(&[ScoreType::Triple]) < e(&[ScoreType::Tetris]));
        // A T-Spin reads as a bigger deal than a plain Triple.
        assert!(e(&[ScoreType::Triple]) < e(&[ScoreType::TSpin]));
    }

    #[test]
    fn combos_and_back_to_back_add_excitement() {
        assert!(
            excitement(&[ScoreType::Tetris, ScoreType::BackToBack])
                > excitement(&[ScoreType::Tetris])
        );
        // Deeper combos escalate.
        assert!(
            excitement(&[ScoreType::Single, ScoreType::Combo(1)])
                < excitement(&[ScoreType::Single, ScoreType::Combo(5)])
        );
    }

    #[test]
    fn style_escalates_then_saturates() {
        let single = callout_style(&[ScoreType::Single]);
        let tetris = callout_style(&[ScoreType::Tetris]);
        let huge = callout_style(&[
            ScoreType::Tetris,
            ScoreType::BackToBack,
            ScoreType::Combo(6),
        ]);

        // Bigger wins => bigger font, more shake, harder pop, longer linger.
        assert!(single.font_size < tetris.font_size);
        assert!(single.shake < tetris.shake);
        assert!(single.pop < tetris.pop);
        assert!(single.ttl < tetris.ttl);

        // A lone Single sits at the calm floor (no shake, base font).
        assert!((single.font_size - CALLOUT_BASE_FONT).abs() < 1e-4);
        assert!(single.shake.abs() < 1e-6);

        // The biggest wins saturate at the ceiling rather than overshoot it.
        assert!((huge.font_size - CALLOUT_MAX_FONT).abs() < 1e-4);
        assert!((huge.ttl - CALLOUT_MAX_TTL).abs() < 1e-4);
        assert!(huge.shake >= tetris.shake);
    }

    #[test]
    fn color_keys_to_the_signature_clear_type() {
        let neutral = Color::srgb(0.94, 0.94, 0.94);
        // T-Spin claims the colour even next to its line label.
        assert_eq!(
            dominant_color(&[ScoreType::TSpin, ScoreType::Double]),
            Color::srgb(0.80, 0.47, 0.97)
        );
        assert_eq!(
            dominant_color(&[ScoreType::Tetris]),
            Color::srgb(0.36, 0.84, 1.0)
        );
        // Every ordinary line clear shares the one neutral colour now; size and
        // shake carry the escalation between them.
        assert_eq!(dominant_color(&[ScoreType::Single]), neutral);
        assert_eq!(dominant_color(&[ScoreType::Double]), neutral);
        assert_eq!(dominant_color(&[ScoreType::Triple]), neutral);
    }
}
