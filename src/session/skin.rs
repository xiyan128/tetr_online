//! The programmatic mino skin: piece-connected, woven, pixel-rounded.
//!
//! Descended from the original `assets/textures/block_tile.png` (dark border
//! around a lighter body) but repainted in code at 32×32 with the texture
//! living at the PIECE level, never as a per-cell motif:
//!
//! * **edge** — the base color slightly darkened, drawn only on EXPOSED
//!   sides (where the neighbor is empty or a different kind). Cells of one
//!   piece share a single perimeter, so a tetromino reads as one designed
//!   object while two touching pieces keep a mortar seam between them.
//! * **weave** — 2×2 dots of darker body tone on an 8 px twill diagonal.
//!   The period divides the cell, so the grain tiles seamlessly over a
//!   whole piece: cloth-like material, the same grain vocabulary as the
//!   ambient background — texture, not ornament. Garbage is weave-LESS:
//!   dead weight has no nap.
//! * **rounded corners** — where two sides open onto EMPTY board, the
//!   outermost 2×2 texels are cut to transparent, pixel-rounding the
//!   silhouette against air. A side facing ANY mino — same piece or not —
//!   stays square, so touching pieces bind flush: mortar seams between
//!   them, never pinholes of background.
//!
//! Every achievable (kind, same-kind mask, open-corner set) is painted once
//! at startup into a [`MinoSkin`] resource; a cell is then a single textured
//! sprite. Hard pixels only (the global nearest sampler), no alpha edges, no
//! glow.

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
/// Tone offsets. The edge is deliberately SOFTER than the original tile's
/// −22% border — at −12% it reads as shading on the tile itself, so the body
/// fills its whole square instead of floating inside a dark recess. The
/// weave sits below the edge tone: quiet cloth tooth that registers at play
/// scale without ever forming a shape.
const EDGE_DARKEN: f32 = 0.12;
const WEAVE_DARKEN: f32 = 0.09;

/// Weave geometry: a [`WEAVE_DOT_PX`]² dot at the origin of every
/// [`WEAVE_PERIOD`]-px tile plus one at its center — a twill diagonal, the
/// same coverage as the old single-texel 4 px grain but chunky enough to
/// read at play scale. The period must divide the cell for the grain to run
/// unbroken across a connected piece.
const WEAVE_PERIOD: usize = 8;
const WEAVE_DOT_PX: usize = 2;
const _: () = assert!(MINO_TEXTURE_SIZE.is_multiple_of(WEAVE_PERIOD));
const _: () = assert!(WEAVE_DOT_PX <= WEAVE_PERIOD / 2);

/// Neighbor-mask bits: a set bit means "same kind continues that way", and
/// that side is painted seamless instead of edged.
pub const MASK_N: u8 = 1;
pub const MASK_E: u8 = 2;
pub const MASK_S: u8 = 4;
pub const MASK_W: u8 = 8;

/// Silhouette corner rounding, in texels: where two sides open onto empty
/// board, this square at the outermost corner is cut to transparent.
const CORNER_CUT_PX: usize = 2;

/// Corner bits (the second texture axis), named for the two sides that meet
/// there in board space.
const CORNER_NE: u8 = 1;
const CORNER_SE: u8 = 2;
const CORNER_SW: u8 = 4;
const CORNER_NW: u8 = 8;

/// The corners whose BOTH flanking sides are set in `sides`. Fed the
/// empty-neighbor mask this is exactly "corners that open onto air" (and so
/// get cut); the builder also feeds it the not-same-kind mask to enumerate
/// which corner sets are achievable for a given connection mask.
fn open_corners(sides: u8) -> u8 {
    let open = |pair: u8, corner: u8| if sides & pair == pair { corner } else { 0 };
    open(MASK_N | MASK_E, CORNER_NE)
        | open(MASK_S | MASK_E, CORNER_SE)
        | open(MASK_S | MASK_W, CORNER_SW)
        | open(MASK_N | MASK_W, CORNER_NW)
}

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

/// Slots per kind: 16 same-kind masks × 16 corner sets (unachievable corner
/// sets alias their clamped neighbor, so every index resolves).
const SLOTS_PER_KIND: usize = 16 * 16;

/// Every painted mino texture, keyed by kind, same-kind connection mask, and
/// the set of corners opening onto empty board. Built once at startup.
#[derive(Resource)]
pub struct MinoSkin {
    handles: Vec<Handle<Image>>,
}

impl MinoSkin {
    /// The texture for a cell: `kind_mask` marks sides where the SAME kind
    /// continues (seamless), `empty_mask` marks sides with no mino at all.
    /// Sides in neither mask hold a different kind: edged like air, but the
    /// corners there stay square so the stack binds without pinholes.
    pub fn handle(&self, kind: MinoKind, kind_mask: u8, empty_mask: u8) -> Handle<Image> {
        debug_assert_eq!(
            kind_mask & empty_mask,
            0,
            "a same-kind side cannot be empty"
        );
        let corners = open_corners(empty_mask & !kind_mask & 0xF);
        let slot = (kind_mask & 0xF) as usize * 16 + corners as usize;
        self.handles[kind.index() * SLOTS_PER_KIND + slot].clone()
    }
}

/// Paint the full skin at startup. Headless test apps run without an asset
/// store; they get placeholder handles (nothing renders there anyway).
pub fn build_mino_skin(mut commands: Commands, images: Option<ResMut<Assets<Image>>>) {
    let Some(mut images) = images else {
        commands.insert_resource(MinoSkin {
            handles: vec![Handle::default(); MinoKind::ALL.len() * SLOTS_PER_KIND],
        });
        return;
    };
    let mut handles = Vec::with_capacity(MinoKind::ALL.len() * SLOTS_PER_KIND);
    for kind in MinoKind::ALL {
        for mask in 0..16u8 {
            // A corner can only open onto air where neither flanking side
            // continues the piece; paint each achievable corner set once and
            // alias the rest onto their clamped set.
            let free = open_corners(!mask & 0xF);
            let mut painted: [Option<Handle<Image>>; 16] = std::array::from_fn(|_| None);
            for corners in 0..16u8 {
                let clamped = (corners & free) as usize;
                let handle = painted[clamped]
                    .get_or_insert_with(|| images.add(paint_mino(kind, mask, clamped as u8)));
                handles.push(handle.clone());
            }
        }
    }
    commands.insert_resource(MinoSkin { handles });
}

/// One mino texture for a kind + connection mask + open-corner set.
fn paint_mino(kind: MinoKind, mask: u8, corners: u8) -> Image {
    let pixels = paint_mino_pixels(kind.base_color(), mask, corners, kind != MinoKind::Garbage);
    debug_assert_eq!(pixels.len(), MINO_TEXTURE_SIZE * MINO_TEXTURE_SIZE * 4);
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

/// The pure painter: edge tone on exposed sides, a seamless 4 px weave over
/// the body (live pieces only), and transparent cuts at the given open
/// corners. Returns RGBA bytes, row 0 at the TOP of the texture (sprite v
/// axis) — callers pass masks in board space, so north (`MASK_N`, y+1 on the
/// board) is the top edge here.
fn paint_mino_pixels(base: [u8; 3], mask: u8, corners: u8, woven: bool) -> Vec<u8> {
    let size = MINO_TEXTURE_SIZE;
    let edge = shade(base, -EDGE_DARKEN);
    let weave = shade(base, -WEAVE_DARKEN);
    let exposed_n = mask & MASK_N == 0;
    let exposed_s = mask & MASK_S == 0;
    let exposed_w = mask & MASK_W == 0;
    let exposed_e = mask & MASK_E == 0;
    let mut pixels = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            // Silhouette rounding: cut the outermost corner block at every
            // corner that opens onto empty board.
            let near_n = y < CORNER_CUT_PX;
            let near_s = y >= size - CORNER_CUT_PX;
            let near_w = x < CORNER_CUT_PX;
            let near_e = x >= size - CORNER_CUT_PX;
            let cut = (corners & CORNER_NW != 0 && near_n && near_w)
                || (corners & CORNER_NE != 0 && near_n && near_e)
                || (corners & CORNER_SW != 0 && near_s && near_w)
                || (corners & CORNER_SE != 0 && near_s && near_e);
            if cut {
                pixels.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let on_edge = (exposed_n && y < EDGE_PX)
                || (exposed_s && y >= size - EDGE_PX)
                || (exposed_w && x < EDGE_PX)
                || (exposed_e && x >= size - EDGE_PX);
            // The weave's period divides the cell, so the grain runs
            // unbroken across every cell of a connected piece.
            let (wx, wy) = (x % WEAVE_PERIOD, y % WEAVE_PERIOD);
            let half = WEAVE_PERIOD / 2;
            let dot = |w: usize, at: usize| w >= at && w < at + WEAVE_DOT_PX;
            let on_weave =
                woven && ((dot(wx, 0) && dot(wy, 0)) || (dot(wx, half) && dot(wy, half)));
            let tone = if on_edge {
                edge
            } else if on_weave {
                weave
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
        let isolated = paint_mino_pixels(base, 0, 0, true);
        let connected_north = paint_mino_pixels(base, MASK_N, 0, true);
        // Isolated: the top row is the edge tone, darker than the body.
        assert_eq!(texel(&isolated, 16, 0), shade(base, -EDGE_DARKEN));
        // Connected to the north: the top row continues the body seamlessly
        // (probe off the weave dots: x % WEAVE_PERIOD == 2, y == 0).
        assert_eq!(texel(&connected_north, 18, 0), base);
        // The other three sides stay edged either way.
        assert_eq!(
            texel(&connected_north, 16, MINO_TEXTURE_SIZE - 1),
            shade(base, -EDGE_DARKEN)
        );
    }

    #[test]
    fn the_weave_is_seamless_across_cells_and_garbage_has_none() {
        let base = [120, 120, 120];
        // Fully connected cell: pure body + weave, no edges, no cuts.
        let pixels = paint_mino_pixels(base, 0xF, 0, true);
        let weave = shade(base, -WEAVE_DARKEN);
        // A 2×2 dot at the tile origin and another at its center…
        assert_eq!(texel(&pixels, 8, 8), weave);
        assert_eq!(texel(&pixels, 9, 9), weave);
        assert_eq!(texel(&pixels, 12, 12), weave);
        assert_eq!(texel(&pixels, 13, 13), weave);
        // …and bare body between them.
        assert_eq!(texel(&pixels, 10, 10), base);
        // The period divides the 32 px cell, so the pattern at one cell's
        // last column continues at the next cell's first column: the texel
        // pattern is purely position-mod-period, identical across the
        // boundary.
        assert_eq!(texel(&pixels, 0, 0), texel(&pixels, 24, 24));
        assert_eq!(texel(&pixels, 2, 2), texel(&pixels, 26, 26));
        // Garbage has no nap.
        let garbage = paint_mino_pixels(base, 0xF, 0, false);
        assert_eq!(texel(&garbage, 8, 8), base);
    }

    #[test]
    fn corners_open_only_where_both_flanking_sides_are_open() {
        assert_eq!(open_corners(0), 0);
        // One open side alone opens no corner.
        assert_eq!(open_corners(MASK_N), 0);
        assert_eq!(open_corners(MASK_N | MASK_E), CORNER_NE);
        assert_eq!(open_corners(MASK_S | MASK_W), CORNER_SW);
        assert_eq!(
            open_corners(0xF),
            CORNER_NE | CORNER_SE | CORNER_SW | CORNER_NW
        );
    }

    #[test]
    fn corners_cut_against_air_but_bind_to_neighboring_pieces() {
        let base = [120, 120, 120];
        let alpha = |pixels: &[u8], x: usize, y: usize| pixels[(y * MINO_TEXTURE_SIZE + x) * 4 + 3];
        let top = 0;
        let bottom = MINO_TEXTURE_SIZE - 1;
        // A lone cell in open air: every corner rounds.
        let lone = paint_mino_pixels(base, 0, open_corners(0xF), true);
        assert_eq!(alpha(&lone, 0, top), 0);
        assert_eq!(alpha(&lone, bottom, bottom), 0);
        // A DIFFERENT piece sits to the north (not same-kind, not empty):
        // both top corners square off and bind flush; the airy bottom still
        // rounds, and the seam side keeps its mortar edge.
        let bound = paint_mino_pixels(base, 0, open_corners(MASK_E | MASK_S | MASK_W), true);
        assert_eq!(alpha(&bound, 0, top), 0xFF);
        assert_eq!(alpha(&bound, bottom, top), 0xFF);
        assert_eq!(alpha(&bound, 0, bottom), 0);
        assert_eq!(texel(&bound, 16, top), shade(base, -EDGE_DARKEN));
        // Same kind continuing east: the shared side is seamless and its two
        // corners stay square; the far (western) corners round into the air.
        let joined = paint_mino_pixels(base, MASK_E, open_corners(MASK_N | MASK_S | MASK_W), true);
        assert_eq!(alpha(&joined, bottom, top), 0xFF);
        assert_eq!(alpha(&joined, 0, top), 0);
        // Interior cell of a piece: no cuts anywhere.
        let interior = paint_mino_pixels(base, 0xF, 0, true);
        assert_eq!(alpha(&interior, 0, top), 0xFF);
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
