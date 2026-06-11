//! The programmatic mino skin: piece-connected, three-tone, faceted.
//!
//! A faithful hi-bit recreation of the original `assets/textures/block_tile.png`
//! (an 8×8 three-tone tile: dark border, mid body, and a bright eight-point
//! faceted core), repainted in code at 32×32 so the tones derive from the
//! Kissaten piece palette instead of a grayscale multiply:
//!
//! * **edge** — the base color darkened, drawn only on EXPOSED sides (where
//!   the neighbor is empty or a different kind). Cells of one piece share a
//!   single perimeter, so a tetromino reads as one designed object while two
//!   touching pieces keep a clear mortar seam between them.
//! * **body** — the muted Kissaten piece color, flat.
//! * **core** — the original's faceted diamond, scaled 4×, slightly lifted.
//!   One facet per cell keeps every mino countable at speed even inside a
//!   merged piece. Garbage is deliberately facet-LESS: dead weight doesn't
//!   sparkle.
//!
//! All 8 kinds × 16 neighbor masks are painted once at startup into a
//! [`MinoSkin`] resource; a cell is then a single textured sprite. Hard
//! pixels only (the global nearest sampler), no alpha edges, no glow.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::engine::PieceType;
use crate::level::common::piece_color;
use crate::ui::widgets::theme;

/// Texture side, px. Four times the original 8 px tile.
pub const MINO_TEXTURE_SIZE: usize = 32;
/// Exposed-edge thickness, px (the original's 1 px border at tile scale,
/// halved for the hi-bit size so the seam stays fine).
const EDGE_PX: usize = 2;
/// Tone offsets. The core keeps the original tile's +7%-ish lift; the edge
/// is deliberately SOFTER than the original's −22% border — at −12% it reads
/// as shading on the tile itself, so the body fills its whole square instead
/// of floating as a rectangle inside a dark recess.
const EDGE_DARKEN: f32 = 0.12;
const CORE_LIGHTEN: f32 = 0.10;

/// Neighbor-mask bits: a set bit means "same kind continues that way", and
/// that side is painted seamless instead of edged.
pub const MASK_N: u8 = 1;
pub const MASK_E: u8 = 2;
pub const MASK_S: u8 = 4;
pub const MASK_W: u8 = 8;

/// The original tile's faceted core, row by row (top to bottom): an
/// eight-point diamond. `1` bits are core texels at 8×8 scale.
const FACET_ROWS: [u8; 8] = [
    0b0000_0000,
    0b0010_0100,
    0b0001_1000,
    0b0011_1100,
    0b0011_1100,
    0b0001_1000,
    0b0010_0100,
    0b0000_0000,
];

/// A paintable cell kind: the seven pieces, or garbage.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum MinoKind {
    Piece(PieceType),
    Garbage,
}

impl MinoKind {
    const ALL: [MinoKind; 8] = [
        MinoKind::Piece(PieceType::I),
        MinoKind::Piece(PieceType::J),
        MinoKind::Piece(PieceType::L),
        MinoKind::Piece(PieceType::O),
        MinoKind::Piece(PieceType::S),
        MinoKind::Piece(PieceType::T),
        MinoKind::Piece(PieceType::Z),
        MinoKind::Garbage,
    ];

    fn index(self) -> usize {
        match self {
            MinoKind::Piece(PieceType::I) => 0,
            MinoKind::Piece(PieceType::J) => 1,
            MinoKind::Piece(PieceType::L) => 2,
            MinoKind::Piece(PieceType::O) => 3,
            MinoKind::Piece(PieceType::S) => 4,
            MinoKind::Piece(PieceType::T) => 5,
            MinoKind::Piece(PieceType::Z) => 6,
            MinoKind::Garbage => 7,
        }
    }

    fn base_color(self) -> [u8; 3] {
        let color = match self {
            MinoKind::Piece(piece) => piece_color(piece),
            MinoKind::Garbage => theme::GARBAGE,
        };
        let srgba = color.to_srgba();
        [
            (srgba.red * 255.0).round() as u8,
            (srgba.green * 255.0).round() as u8,
            (srgba.blue * 255.0).round() as u8,
        ]
    }
}

/// Every painted mino texture: 8 kinds × 16 neighbor masks, built once.
#[derive(Resource)]
pub struct MinoSkin {
    handles: Vec<Handle<Image>>,
}

impl MinoSkin {
    pub fn handle(&self, kind: MinoKind, mask: u8) -> Handle<Image> {
        self.handles[kind.index() * 16 + (mask & 0xF) as usize].clone()
    }
}

/// Paint the full skin at startup. Headless test apps run without an asset
/// store; they get placeholder handles (nothing renders there anyway).
pub fn build_mino_skin(mut commands: Commands, images: Option<ResMut<Assets<Image>>>) {
    let Some(mut images) = images else {
        commands.insert_resource(MinoSkin {
            handles: vec![Handle::default(); MinoKind::ALL.len() * 16],
        });
        return;
    };
    let mut handles = Vec::with_capacity(MinoKind::ALL.len() * 16);
    for kind in MinoKind::ALL {
        for mask in 0..16u8 {
            handles.push(images.add(paint_mino(kind, mask)));
        }
    }
    commands.insert_resource(MinoSkin { handles });
}

/// One mino texture for a kind + neighbor mask, as an RGBA image.
fn paint_mino(kind: MinoKind, mask: u8) -> Image {
    let pixels = paint_mino_pixels(kind.base_color(), mask, kind != MinoKind::Garbage);
    Image::new(
        Extent3d {
            width: MINO_TEXTURE_SIZE as u32,
            height: MINO_TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// The pure painter: edge tone on exposed sides, flat body, and (for live
/// pieces) the original's faceted core scaled 4×. Returns RGBA bytes, row 0
/// at the TOP of the texture (sprite v axis) — callers pass a mask in board
/// space, so north (`MASK_N`, y+1 on the board) is the top edge here.
fn paint_mino_pixels(base: [u8; 3], mask: u8, faceted: bool) -> Vec<u8> {
    let size = MINO_TEXTURE_SIZE;
    let edge = shade(base, -EDGE_DARKEN);
    let core = shade(base, CORE_LIGHTEN);
    let mut pixels = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let exposed_n = mask & MASK_N == 0 && y < EDGE_PX;
            let exposed_s = mask & MASK_S == 0 && y >= size - EDGE_PX;
            let exposed_w = mask & MASK_W == 0 && x < EDGE_PX;
            let exposed_e = mask & MASK_E == 0 && x >= size - EDGE_PX;
            let tone = if exposed_n || exposed_s || exposed_w || exposed_e {
                edge
            } else if faceted && FACET_ROWS[y / 4] & (0x80 >> (x / 4)) != 0 {
                core
            } else {
                base
            };
            pixels.extend_from_slice(&[tone[0], tone[1], tone[2], 0xFF]);
        }
    }
    pixels
}

/// Shade an sRGB color: positive `amount` lightens toward white, negative
/// darkens toward black, both proportionally (the original tile's tones are
/// ratios of its body, not fixed offsets).
fn shade(color: [u8; 3], amount: f32) -> [u8; 3] {
    std::array::from_fn(|i| {
        let channel = color[i] as f32;
        let shaded = if amount >= 0.0 {
            channel + (255.0 - channel) * amount
        } else {
            channel * (1.0 + amount)
        };
        shaded.round().clamp(0.0, 255.0) as u8
    })
}

/// Neighbor mask from an arbitrary sameness predicate — the locked board
/// uses it to merge same-kind neighbors out of a keyed cell map.
pub fn neighbor_mask_where(x: isize, y: isize, same: impl Fn(isize, isize) -> bool) -> u8 {
    let mut mask = 0;
    if same(x, y + 1) {
        mask |= MASK_N;
    }
    if same(x + 1, y) {
        mask |= MASK_E;
    }
    if same(x, y - 1) {
        mask |= MASK_S;
    }
    if same(x - 1, y) {
        mask |= MASK_W;
    }
    mask
}

/// Neighbor mask for `cell` against a set of same-kind occupied positions.
pub fn neighbor_mask(
    x: isize,
    y: isize,
    occupied: &std::collections::HashSet<(isize, isize)>,
) -> u8 {
    let mut mask = 0;
    if occupied.contains(&(x, y + 1)) {
        mask |= MASK_N;
    }
    if occupied.contains(&(x + 1, y)) {
        mask |= MASK_E;
    }
    if occupied.contains(&(x, y - 1)) {
        mask |= MASK_S;
    }
    if occupied.contains(&(x - 1, y)) {
        mask |= MASK_W;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texel(pixels: &[u8], x: usize, y: usize) -> [u8; 3] {
        let at = (y * MINO_TEXTURE_SIZE + x) * 4;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    }

    #[test]
    fn exposed_edges_are_darker_and_connected_edges_are_seamless() {
        let base = [200, 150, 100];
        let isolated = paint_mino_pixels(base, 0, true);
        let connected_north = paint_mino_pixels(base, MASK_N, true);
        // Isolated: the top row is the edge tone, darker than the body.
        assert_eq!(texel(&isolated, 16, 0), shade(base, -EDGE_DARKEN));
        // Connected to the north: the top row continues the body seamlessly.
        assert_eq!(texel(&connected_north, 16, 0), base);
        // The other three sides stay edged either way.
        assert_eq!(
            texel(&connected_north, 16, MINO_TEXTURE_SIZE - 1),
            shade(base, -EDGE_DARKEN)
        );
    }

    #[test]
    fn the_facet_core_matches_the_original_tile() {
        let base = [120, 120, 120];
        let pixels = paint_mino_pixels(base, 0xF, true);
        let core = shade(base, CORE_LIGHTEN);
        // Center of the diamond (original texel 3,3 → 4× block at 12..16).
        assert_eq!(texel(&pixels, 14, 14), core);
        // The corner sparks (original texel 2,1).
        assert_eq!(texel(&pixels, 10, 6), core);
        // Between spark and diamond stays body (original texel 3,1 is mid).
        assert_eq!(texel(&pixels, 14, 6), base);
        // Garbage never sparkles.
        let garbage = paint_mino_pixels(base, 0xF, false);
        assert_eq!(texel(&garbage, 14, 14), base);
    }

    #[test]
    fn neighbor_mask_reads_board_adjacency() {
        let occupied: std::collections::HashSet<(isize, isize)> =
            [(5, 6), (6, 5), (4, 4)].into_iter().collect();
        // North of (5,5) is (5,6); east is (6,5); (4,4) is diagonal — no bit.
        assert_eq!(neighbor_mask(5, 5, &occupied), MASK_N | MASK_E);
        assert_eq!(neighbor_mask(0, 0, &std::collections::HashSet::new()), 0);
    }

    #[test]
    fn shade_is_bounded_and_directional() {
        assert_eq!(shade([255, 255, 255], 0.5), [255, 255, 255]);
        assert_eq!(shade([0, 0, 0], -0.5), [0, 0, 0]);
        let lighter = shade([100, 100, 100], 0.1);
        let darker = shade([100, 100, 100], -0.1);
        assert!(lighter[0] > 100 && darker[0] < 100);
    }
}
