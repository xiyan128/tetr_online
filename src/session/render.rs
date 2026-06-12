//! Session rendering: a board group per seat, everything parented to it.
//!
//! Each seat gets a **board root** entity at its world-space origin; the
//! field chrome (backplate + frame), the mino layers (locked / falling /
//! ghost, all skinned by `session::skin`), the hold and
//! preview columns, the garbage meter, and the seat texts are all children of
//! (or anchored to) that root — position is one transform, despawn is one
//! subtree, and a future mirrored layout is a per-root parameter. One camera
//! frames both boards with `ScalingMode::AutoMin`, so native window resizes
//! and the web canvas keep the whole match visible. The camera carries the
//! [`GameplayCamera`] tag, so the optional bloom skin applies to it; the
//! screen-shake mover (`session::feel`) offsets it around the scene's rest
//! center.
//!
//! Reconcilers diff the cached snapshot slice and despawn-and-respawn only
//! what changed, querying per seat. Garbage cells (`SnapshotCell::garbage`) paint a
//! warm zero-chroma gray: telling your own stack from their attack at a glance is the
//! point of having a versus renderer at all.

use bevy::camera::ScalingMode;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use std::collections::HashMap;

use crate::GameState;
use crate::assets::GameAssets;
use crate::engine::{Piece, PieceType, SnapshotCell};
use crate::level::common::{GameplayCamera, to_translation};
use crate::ui::widgets::theme;

use super::skin::{self, MinoKind, MinoSkin};

use super::{Participant, Seat, SeatSnapshot, SeatStats, SessionConfig};

/// World-space layout of the two-board scene, in cells and pixels. One home
/// for every magic number the renderer and overlays share.
pub struct SessionLayout;

impl SessionLayout {
    pub const BLOCK: f32 = 32.0;
    pub const BOARD_W: usize = 10;
    pub const BOARD_H: usize = 20;
    /// Cells between the two boards — wide enough for seat 0's preview column
    /// and seat 1's hold column (each beyond its garbage-meter channel) to
    /// meet without touching.
    pub const GUTTER_CELLS: f32 = 10.0;
    /// Air on each side of the garbage meter, in cells: the bar floats in
    /// its own channel between the board frame and the inner hold/preview
    /// column, touching neither.
    pub const METER_AIR_CELLS: f32 = 0.5;
    /// Avatar scale for hold/preview minos (matches the single-player feel).
    pub const PREVIEW_SCALE: f32 = 0.8;
    /// Vertical breathing room between the board frame and the run readout
    /// beneath it ("ATK n" / the solo stat line), in cells. Generous on
    /// purpose: the readout is its own quiet row, not a caption glued to the
    /// frame.
    pub const READOUT_GAP_CELLS: f32 = 1.2;

    /// Offset from a board edge to its hold/preview column anchor, in cells.
    /// On a versus board's INNER edge (`beyond_meter`) the column steps out
    /// to clear the garbage-meter channel: air, the meter track, air again.
    pub fn column_gap_cells(beyond_meter: bool) -> f32 {
        if beyond_meter {
            2.0 * Self::METER_AIR_CELLS + METER_WIDTH / Self::BLOCK
        } else {
            0.5
        }
    }

    /// World-space origin (bottom-left cell) of a seat's board.
    pub fn board_origin(seat: usize) -> Vec3 {
        let stride = (Self::BOARD_W as f32 + Self::GUTTER_CELLS) * Self::BLOCK;
        Vec3::new(seat as f32 * stride, 0.0, 0.0)
    }

    /// Center of the whole scene (the camera's rest position), for however
    /// many seats this session plays with.
    pub fn scene_center(seat_count: usize) -> Vec3 {
        let right =
            Self::board_origin(seat_count.saturating_sub(1)).x + Self::BOARD_W as f32 * Self::BLOCK;
        Vec3::new(right / 2.0, Self::BOARD_H as f32 * Self::BLOCK / 2.0, 1.0)
    }

    /// Minimum world-space rectangle the camera must keep visible: every
    /// board, the outer hold/preview columns, the texts above and below, and
    /// one display-size callout word in each OUTER gutter (the callout stack
    /// in `session::feel` is one word per line; Dogica is monospaced at
    /// 1 em/glyph, so the longest word — "BACK-TO-BACK" at 16 px, "TETRIS" at
    /// 32 px — needs ~6 cells outboard of each board edge).
    pub fn scene_min(seat_count: usize) -> (f32, f32) {
        let width_cells = match seat_count {
            // hold column + board + preview column + a callout gutter.
            0 | 1 => 24.0,
            // two board groups, the shared gutter, and a callout gutter on
            // each outer flank.
            _ => 45.0,
        };
        // Height: the 20-cell board plus headroom for the seat label and,
        // below, the readout at its READOUT_GAP_CELLS offset with clear air
        // to the screen edge.
        (width_cells * Self::BLOCK, 26.0 * Self::BLOCK)
    }
}

/// A seat's render anchor; all of the seat's visuals hang off this entity.
/// `seat` is the identity an overlay or a future mirrored-layout pass selects
/// roots by (today's readers use the layer/meter markers, hence the allow).
#[derive(Component)]
pub struct BoardRoot {
    #[allow(dead_code)]
    pub seat: usize,
}

/// Which mino population a layer entity holds.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum VsLayer {
    Static,
    Falling,
    Ghost,
}

/// Marks a layer entity with its seat (paired with [`VsLayer`]).
#[derive(Component, Clone, Copy)]
pub struct LayerSeat(pub usize);

/// World-space width of the garbage meter (track and fill), in pixels.
const METER_WIDTH: f32 = 8.0;

/// The garbage meter root for a seat; children are the batch segments.
#[derive(Component)]
pub struct SeatMeter {
    pub seat: usize,
}

/// One edge of a seat's field frame. The danger pass warms these toward
/// `ATTACK` as the stack nears the top — the frame carries the state so the
/// field background never has to change during play.
#[derive(Component)]
pub struct BoardFrame {
    pub seat: usize,
}

/// The hold avatar container for a seat.
#[derive(Component)]
pub struct SeatHoldView {
    pub seat: usize,
}

/// The next-queue avatar container for a seat. The render cache lives ON the
/// component (not a system `Local`) so it dies with the session — a `Local`
/// would survive into the next match and skip the first rebuild whenever a
/// pinned seed re-deals the identical opening queue.
#[derive(Component)]
pub struct SeatPreviewView {
    pub seat: usize,
    cache: Option<Vec<PieceType>>,
}

/// The cumulative-attack readout under a seat's board.
#[derive(Component)]
pub struct SeatAtkText {
    pub seat: usize,
}

pub struct SessionRenderPlugin;

impl Plugin for SessionRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, skin::build_mino_skin)
            .add_systems(OnEnter(GameState::Session), setup_scene)
            .add_systems(
                Update,
                (
                    reconcile_locked_boards,
                    reconcile_active_pieces,
                    reconcile_ghost_pieces,
                    reconcile_garbage_meters,
                    reconcile_hold_views,
                    reconcile_preview_views,
                    update_atk_texts,
                    update_seat_timer_bars,
                    tint_danger_frames,
                )
                    .run_if(in_state(GameState::Session)),
            );
    }
}

/// Ghost-outline stroke thickness as a fraction of the cell — matches the
/// skin's exposed-edge seam so the outline speaks the same pixel dialect.
const GHOST_STROKE_FRACTION: f32 = 2.0 / 32.0;

/// One textured mino at board-relative cell `(x, y)`: a single sprite over
/// the painted skin (`session::skin`), full cell size — adjacency is drawn
/// INTO the texture (the neighbor mask), not left as a gap. On opt-in bloom
/// builds the sprite tint lifts the texture into HDR for the glow.
fn block_sprite(
    image: Handle<Image>,
    block_size: f32,
    x: isize,
    y: isize,
    z: f32,
) -> impl Bundle + use<> {
    let mut sprite = Sprite::from_image(image);
    sprite.custom_size = Some(Vec2::splat(block_size));
    #[cfg(feature = "bloom")]
    {
        sprite.color = Color::srgb(1.6, 1.6, 1.6);
    }
    let mut transform = Transform::from_translation(to_translation(x, y, block_size));
    transform.translation.z = z;
    (sprite, transform, Anchor::BOTTOM_LEFT)
}

/// The skin kind of a snapshot cell: garbage, or its piece.
fn cell_kind(cell: &SnapshotCell) -> MinoKind {
    if cell.garbage {
        MinoKind::Garbage
    } else {
        MinoKind::Piece(cell.piece_type)
    }
}

/// Spawn the static scene: per-seat roots (grid, layers, meter, hold/preview
/// containers, texts) and the camera framing both boards.
fn setup_scene(
    mut commands: Commands,
    assets: Res<GameAssets>,
    config: Res<SessionConfig>,
    registry: Res<crate::ai::ModelRegistry>,
) {
    let block = SessionLayout::BLOCK;

    for seat in 0..config.mode.seat_count() {
        let origin = SessionLayout::board_origin(seat);
        let root = commands
            .spawn((
                BoardRoot { seat },
                Transform::from_translation(origin),
                Visibility::default(),
                DespawnOnExit(GameState::Session),
            ))
            .id();

        // Field chrome, drawn once: an opaque `BG` backplate (the board
        // interior is exactly the ground — it blocks the ambient background
        // layer and anchors "the field never changes") and a 1 px `FRAME`
        // border just outside the field. No gridlines: the field is calm
        // open ground, and the mino skin's connected edges carry all the
        // articulation.
        let board_w = SessionLayout::BOARD_W as f32 * block;
        let board_h = SessionLayout::BOARD_H as f32 * block;
        let mut chrome = Vec::new();
        chrome.push(
            commands
                .spawn((
                    Sprite::from_color(theme::BG, Vec2::new(board_w, board_h)),
                    Transform::from_translation(Vec3::new(0.0, 0.0, -2.0)),
                    Anchor::BOTTOM_LEFT,
                ))
                .id(),
        );
        // The frame's four edges carry `BoardFrame` so the danger pass can
        // warm them toward `ATTACK` as the stack climbs.
        let frame_edges = [
            (Vec2::new(-1.0, -1.0), Vec2::new(board_w + 2.0, 1.0)),
            (Vec2::new(-1.0, board_h), Vec2::new(board_w + 2.0, 1.0)),
            (Vec2::new(-1.0, 0.0), Vec2::new(1.0, board_h)),
            (Vec2::new(board_w, 0.0), Vec2::new(1.0, board_h)),
        ];
        for (pos, size) in frame_edges {
            chrome.push(
                commands
                    .spawn((
                        BoardFrame { seat },
                        Sprite::from_color(theme::FRAME, size),
                        Transform::from_translation(pos.extend(-0.9)),
                        Anchor::BOTTOM_LEFT,
                    ))
                    .id(),
            );
        }
        commands.entity(root).add_children(&chrome);

        // Mino layers: children rebuilt by the reconcilers.
        for layer in [VsLayer::Static, VsLayer::Falling, VsLayer::Ghost] {
            let id = commands
                .spawn((
                    layer,
                    LayerSeat(seat),
                    Transform::default(),
                    Visibility::default(),
                ))
                .id();
            commands.entity(root).add_child(id);
        }

        // Garbage meter on the board's INNER edge (facing the opponent):
        // seat 0's on its right, seat 1's on its left. A full-height track in
        // `GRID` reads as a quiet groove at rest; the fill segments (children
        // of the meter root, one per pending batch) arrive in `ATTACK` and
        // stack upward from the board floor. The bar floats with
        // `METER_AIR_CELLS` of air on BOTH sides — frame · air · meter ·
        // air · column — never glued to either. Versus only — solo has no
        // garbage channel, and an always-empty groove would read as a stuck
        // UI element (the meter reconciler tolerates the missing root).
        let versus = matches!(config.mode, super::SessionMode::Versus);
        if versus {
            let air = SessionLayout::METER_AIR_CELLS * block;
            let meter_x = if seat == 0 {
                SessionLayout::BOARD_W as f32 * block + air
            } else {
                -air - METER_WIDTH
            };
            let track = commands
                .spawn((
                    Sprite::from_color(theme::GRID, Vec2::new(METER_WIDTH, board_h)),
                    Transform::from_translation(Vec3::new(meter_x, 0.0, 0.4)),
                    Anchor::BOTTOM_LEFT,
                ))
                .id();
            let meter = commands
                .spawn((
                    SeatMeter { seat },
                    Transform::from_translation(Vec3::new(meter_x, 0.0, 0.5)),
                    Visibility::default(),
                ))
                .id();
            commands.entity(root).add_children(&[track, meter]);
        }

        // Lock-down progress bar, under the field (grows left→right toward
        // lock). Quiet chrome: `FRAME`, like every other resting border.
        let bar_height = SessionLayout::BLOCK * 0.2;
        let bar = commands
            .spawn((
                SeatTimerBar { seat },
                Sprite {
                    custom_size: Some(Vec2::new(0.0, bar_height)),
                    color: theme::FRAME,
                    ..Default::default()
                },
                Anchor::BOTTOM_LEFT,
                Transform::from_translation(Vec3::new(0.0, -bar_height, 1.0)),
            ))
            .id();
        commands.entity(root).add_child(bar);

        // Hold column (top-left of the board) and preview column (top-right) —
        // the single-player arrangement, duplicated per seat. On a versus
        // board's inner edge the column steps outboard past the garbage-meter
        // channel so the bar keeps clear air on both sides.
        let hold_gap = SessionLayout::column_gap_cells(versus && seat == 1);
        let preview_gap = SessionLayout::column_gap_cells(versus && seat == 0);
        let hold = commands
            .spawn((
                SeatHoldView { seat },
                Transform::from_translation(Vec3::new(
                    -hold_gap * block,
                    SessionLayout::BOARD_H as f32 * block,
                    0.0,
                )),
                Visibility::default(),
            ))
            .id();
        let preview = commands
            .spawn((
                SeatPreviewView { seat, cache: None },
                Transform::from_translation(Vec3::new(
                    (SessionLayout::BOARD_W as f32 + preview_gap) * block,
                    SessionLayout::BOARD_H as f32 * block,
                    0.0,
                )),
                Visibility::default(),
            ))
            .id();
        // Micro labels over the columns, in the working voice — quiet chrome
        // that makes the HUD legible cold (the columns themselves are unboxed).
        let hold_label = commands
            .spawn((
                Text2d::new("HOLD"),
                TextFont {
                    font: assets.font_body.clone(),
                    font_size: theme::MICRO_FONT_SIZE,
                    ..default()
                },
                TextColor(theme::TEXT_DIM),
                Anchor::BOTTOM_RIGHT,
                Transform::from_translation(Vec3::new(
                    -hold_gap * block,
                    (SessionLayout::BOARD_H as f32 + 0.15) * block,
                    0.0,
                )),
            ))
            .id();
        let next_label = commands
            .spawn((
                Text2d::new("NEXT"),
                TextFont {
                    font: assets.font_body.clone(),
                    font_size: theme::MICRO_FONT_SIZE,
                    ..default()
                },
                TextColor(theme::TEXT_DIM),
                Anchor::BOTTOM_LEFT,
                Transform::from_translation(Vec3::new(
                    (SessionLayout::BOARD_W as f32 + preview_gap) * block,
                    (SessionLayout::BOARD_H as f32 + 0.15) * block,
                    0.0,
                )),
            ))
            .id();
        commands
            .entity(root)
            .add_children(&[hold, preview, hold_label, next_label]);

        // Seat label above the board: "YOU", or the model's catalog name.
        let label = match config.seats[seat] {
            Participant::Human => "YOU".to_string(),
            Participant::Bot { model } => registry.label(model).to_uppercase(),
        };
        let label_id = commands
            .spawn((
                Text2d::new(label),
                TextFont {
                    font: assets.font.clone(),
                    font_size: theme::BUTTON_FONT_SIZE,
                    ..default()
                },
                TextColor(theme::TEXT),
                Anchor::BOTTOM_CENTER,
                Transform::from_translation(Vec3::new(
                    SessionLayout::BOARD_W as f32 * block / 2.0,
                    (SessionLayout::BOARD_H as f32 + 0.6) * block,
                    0.0,
                )),
            ))
            .id();

        // The readout under the board. Versus shows the pure "ATK n" figure at
        // the numeral size (numerals are the hero typography); solo packs the
        // whole run line into one row, so it stays at the label size.
        let readout_size = match config.mode {
            super::SessionMode::Versus => theme::NUMERAL_FONT_SIZE,
            super::SessionMode::Solo { .. } => theme::BUTTON_FONT_SIZE,
        };
        let atk_id = commands
            .spawn((
                SeatAtkText { seat },
                Text2d::new("ATK 0"),
                TextFont {
                    font: assets.font.clone(),
                    font_size: readout_size,
                    ..default()
                },
                TextColor(theme::TEXT),
                Anchor::TOP_CENTER,
                Transform::from_translation(Vec3::new(
                    SessionLayout::BOARD_W as f32 * block / 2.0,
                    -SessionLayout::READOUT_GAP_CELLS * block,
                    0.0,
                )),
            ))
            .id();
        commands.entity(root).add_children(&[label_id, atk_id]);
    }

    // One camera, every board always in frame. `GameplayCamera` opts into the
    // optional effects stack and is the entity the shake mover offsets. It
    // composites over the ambient background pass instead of clearing it.
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
        GameplayCamera,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: {
                let (min_width, min_height) = SessionLayout::scene_min(config.mode.seat_count());
                ScalingMode::AutoMin {
                    min_width,
                    min_height,
                }
            },
            ..OrthographicProjection::default_2d()
        }),
        Transform::from_translation(SessionLayout::scene_center(config.mode.seat_count())),
        DespawnOnExit(GameState::Session),
    ));
}

/// Find the layer entity for `(seat, kind)`.
fn layer_for(
    layers: &Query<(Entity, &VsLayer, &LayerSeat)>,
    seat: usize,
    kind: VsLayer,
) -> Option<Entity> {
    layers
        .iter()
        .find(|(_, l, s)| **l == kind && s.0 == seat)
        .map(|(e, _, _)| e)
}

/// Rebuild a seat's locked minos when its board changes (diffed against the
/// cached cells, exactly like the single-player reconciler).
fn reconcile_locked_boards(
    mut commands: Commands,
    minos: Res<MinoSkin>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    layers: Query<(Entity, &VsLayer, &LayerSeat)>,
    mut cache: Local<[Option<Vec<SnapshotCell>>; 2]>,
) {
    for (seat, snapshot) in &seats {
        let index = seat.index.min(1);
        let cells = &snapshot.0.board_cells;
        if cache[index].as_ref() == Some(cells) {
            continue;
        }
        let Some(layer) = layer_for(&layers, seat.index, VsLayer::Static) else {
            continue;
        };
        commands.entity(layer).despawn_related::<Children>();
        // Connectedness: a cell merges (seamless side) with same-kind
        // neighbors, so a piece shares one perimeter and garbage reads as a
        // slab; different kinds keep the mortar seam between them. Corners
        // round only against truly empty board, so touching pieces pack
        // flush — no pinholes of background inside the stack.
        let kinds: HashMap<(isize, isize), MinoKind> = cells
            .iter()
            .map(|cell| ((cell.x, cell.y), cell_kind(cell)))
            .collect();
        let ids: Vec<Entity> = cells
            .iter()
            .map(|cell| {
                let kind = cell_kind(cell);
                let kind_mask = skin::neighbor_mask_where(cell.x, cell.y, |x, y| {
                    kinds.get(&(x, y)) == Some(&kind)
                });
                let empty_mask = 0xF
                    & !skin::neighbor_mask_where(cell.x, cell.y, |x, y| {
                        kinds.contains_key(&(x, y))
                    });
                commands
                    .spawn(block_sprite(
                        minos.handle(kind, kind_mask, empty_mask),
                        SessionLayout::BLOCK,
                        cell.x,
                        cell.y,
                        0.0,
                    ))
                    .id()
            })
            .collect();
        commands.entity(layer).add_children(&ids);
        cache[index] = Some(cells.clone());
    }
}

/// Rebuild each seat's falling piece every frame (4 sprites; always in sync).
fn reconcile_active_pieces(
    mut commands: Commands,
    minos: Res<MinoSkin>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    layers: Query<(Entity, &VsLayer, &LayerSeat)>,
) {
    for (seat, snapshot) in &seats {
        let Some(layer) = layer_for(&layers, seat.index, VsLayer::Falling) else {
            continue;
        };
        commands.entity(layer).despawn_related::<Children>();
        let Some(active) = snapshot.0.active.as_ref() else {
            continue;
        };
        // The four cells are one piece by definition: they merge into a
        // single connected object with one shared perimeter. In flight the
        // piece is wrapped in air — every exposed corner rounds; it binds to
        // the stack only once it locks.
        let cells = &active.cells;
        let ids: Vec<Entity> = cells
            .iter()
            .map(|cell| {
                let mask = skin::neighbor_mask_where(cell.x, cell.y, |x, y| {
                    cells.iter().any(|c| c.x == x && c.y == y)
                });
                commands
                    .spawn(block_sprite(
                        minos.handle(cell_kind(cell), mask, skin::airborne_empty(mask)),
                        SessionLayout::BLOCK,
                        cell.x,
                        cell.y,
                        0.0,
                    ))
                    .id()
            })
            .collect();
        commands.entity(layer).add_children(&ids);
    }
}

/// The ghost is an outline, never a fill (`theme::TEXT` at 35%): four hairline
/// edges per cell, stroke as thick as the articulation gap. Readable at a
/// glance and never mistakable for a placed mino.
/// Outline edges for one ghost cell, EXPOSED sides only (`mask` as in the
/// mino skin): the ghost traces the piece's connected silhouette, matching
/// the skin's shared-perimeter language, never a per-cell lattice.
fn ghost_cell_outline(
    commands: &mut Commands,
    block: f32,
    x: isize,
    y: isize,
    mask: u8,
) -> Vec<Entity> {
    let stroke = block * GHOST_STROKE_FRACTION;
    let base = to_translation(x, y, block) + Vec3::new(0.0, 0.0, -0.1);
    let color = theme::TEXT.with_alpha(0.35);
    // (mask bit, offset, size): the full side, so silhouette corners meet.
    let edges = [
        (skin::MASK_S, Vec2::new(0.0, 0.0), Vec2::new(block, stroke)),
        (
            skin::MASK_N,
            Vec2::new(0.0, block - stroke),
            Vec2::new(block, stroke),
        ),
        (skin::MASK_W, Vec2::new(0.0, 0.0), Vec2::new(stroke, block)),
        (
            skin::MASK_E,
            Vec2::new(block - stroke, 0.0),
            Vec2::new(stroke, block),
        ),
    ];
    edges
        .into_iter()
        .filter(|(bit, _, _)| mask & bit == 0)
        .map(|(_, offset, size)| {
            commands
                .spawn((
                    Sprite::from_color(color, size),
                    Transform::from_translation(base + offset.extend(0.0)),
                    Anchor::BOTTOM_LEFT,
                ))
                .id()
        })
        .collect()
}

/// Rebuild a seat's ghost when it changes (hidden when grounded or disabled —
/// folded into the cache key as an empty cell list, so the rebuild also
/// happens only on the show/hide transition, not every hidden frame).
fn reconcile_ghost_pieces(
    mut commands: Commands,
    settings: Res<crate::settings::GameSettings>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    layers: Query<(Entity, &VsLayer, &LayerSeat)>,
    mut cache: Local<[Option<Vec<SnapshotCell>>; 2]>,
) {
    for (seat, snapshot) in &seats {
        let index = seat.index.min(1);
        let airborne = snapshot
            .0
            .active
            .as_ref()
            .is_some_and(|active| !active.landed);
        let cells: &[SnapshotCell] = if settings.ghost_enabled && airborne {
            &snapshot.0.ghost_cells
        } else {
            &[]
        };
        if cache[index].as_deref() == Some(cells) {
            continue;
        }
        let Some(layer) = layer_for(&layers, seat.index, VsLayer::Ghost) else {
            continue;
        };
        commands.entity(layer).despawn_related::<Children>();
        let ids: Vec<Entity> = cells
            .iter()
            .flat_map(|cell| {
                let mask = skin::neighbor_mask_where(cell.x, cell.y, |x, y| {
                    cells.iter().any(|c| c.x == x && c.y == y)
                });
                ghost_cell_outline(&mut commands, SessionLayout::BLOCK, cell.x, cell.y, mask)
            })
            .collect();
        commands.entity(layer).add_children(&ids);
        cache[index] = Some(cells.to_vec());
    }
}

/// A pending batch as the meter draws it: `(lines, hole_col)`.
type MeterBatches = Vec<(u32, usize)>;

/// Rebuild a seat's garbage meter when its pending queue changes: one red
/// segment per batch, stacked bottom-up in arrival order with a small notch
/// between batches — "a 4 and a 2 are coming" reads at a glance.
fn reconcile_garbage_meters(
    mut commands: Commands,
    seats: Query<(&Seat, &SeatSnapshot)>,
    meters: Query<(Entity, &SeatMeter)>,
    mut cache: Local<[Option<MeterBatches>; 2]>,
) {
    const NOTCH: f32 = 4.0;
    for (seat, snapshot) in &seats {
        let index = seat.index.min(1);
        let batches: MeterBatches = snapshot
            .0
            .pending_garbage
            .iter()
            .map(|b| (b.lines, b.hole_col))
            .collect();
        if cache[index].as_ref() == Some(&batches) {
            continue;
        }
        let Some((meter, _)) = meters.iter().find(|(_, m)| m.seat == seat.index) else {
            continue;
        };
        commands.entity(meter).despawn_related::<Children>();
        let mut y = 0.0;
        let mut ids = Vec::new();
        for (lines, _) in &batches {
            let height = *lines as f32 * SessionLayout::BLOCK - NOTCH;
            let sprite = Sprite::from_color(
                crate::ui::widgets::theme::ATTACK,
                Vec2::new(METER_WIDTH, height.max(NOTCH)),
            );
            ids.push(
                commands
                    .spawn((
                        sprite,
                        Transform::from_translation(Vec3::new(0.0, y, 0.0)),
                        Anchor::BOTTOM_LEFT,
                    ))
                    .id(),
            );
            y += *lines as f32 * SessionLayout::BLOCK;
        }
        commands.entity(meter).add_children(&ids);
        cache[index] = Some(batches);
    }
}

/// Spawn a piece avatar (the hold/preview mino cluster) under `parent`,
/// anchored at the parent's origin and growing down-right. Returns the
/// avatar's world height so callers can stack entries.
fn spawn_avatar(
    commands: &mut Commands,
    minos: &MinoSkin,
    parent: Entity,
    piece_type: PieceType,
    y_top: f32,
    align_right: bool,
) -> f32 {
    let piece = Piece::from(piece_type);
    let block = SessionLayout::BLOCK * SessionLayout::PREVIEW_SCALE;
    let (avatar_w, avatar_h) = piece.avatar_dims();
    let height = avatar_h as f32 * block;
    let x_off = if align_right {
        -(avatar_w as f32) * block
    } else {
        0.0
    };

    let holder = commands
        .spawn((
            Transform::from_translation(Vec3::new(x_off, y_top - height, 0.0)),
            Visibility::default(),
        ))
        .id();
    // The avatar is one piece: connected, one shared perimeter, air all
    // around (every exposed corner rounds).
    let cells = piece.avatar_cells();
    let ids: Vec<Entity> = cells
        .iter()
        .map(|&(x, y)| {
            let mask = skin::neighbor_mask_where(x, y, |nx, ny| cells.contains(&(nx, ny)));
            commands
                .spawn(block_sprite(
                    minos.handle(
                        MinoKind::Piece(piece_type),
                        mask,
                        skin::airborne_empty(mask),
                    ),
                    block,
                    x,
                    y,
                    0.0,
                ))
                .id()
        })
        .collect();
    commands.entity(holder).add_children(&ids);
    commands.entity(parent).add_child(holder);
    height
}

/// Rebuild a seat's hold avatar when the held piece changes.
fn reconcile_hold_views(
    mut commands: Commands,
    minos: Res<MinoSkin>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    views: Query<(Entity, &SeatHoldView)>,
    mut cache: Local<[Option<Option<PieceType>>; 2]>,
) {
    for (seat, snapshot) in &seats {
        let index = seat.index.min(1);
        let hold = snapshot.0.hold;
        if cache[index] == Some(hold) {
            continue;
        }
        let Some((view, _)) = views.iter().find(|(_, v)| v.seat == seat.index) else {
            continue;
        };
        commands.entity(view).despawn_related::<Children>();
        if let Some(piece_type) = hold {
            spawn_avatar(&mut commands, &minos, view, piece_type, 0.0, true);
        }
        cache[index] = Some(hold);
    }
}

/// Rebuild a seat's next-queue column when the visible queue changes.
fn reconcile_preview_views(
    mut commands: Commands,
    minos: Res<MinoSkin>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    mut views: Query<(Entity, &mut SeatPreviewView)>,
) {
    for (seat, snapshot) in &seats {
        let queue = &snapshot.0.next_queue;
        let Some((view, mut state)) = views.iter_mut().find(|(_, v)| v.seat == seat.index) else {
            continue;
        };
        if state.cache.as_ref() == Some(queue) {
            continue;
        }
        commands.entity(view).despawn_related::<Children>();
        let gap = 0.5 * SessionLayout::BLOCK * SessionLayout::PREVIEW_SCALE;
        let mut y_top = 0.0;
        for &piece_type in queue {
            let height = spawn_avatar(&mut commands, &minos, view, piece_type, y_top, false);
            y_top -= height + gap;
        }
        state.cache = Some(queue.clone());
    }
}

/// Keep each seat's under-board readout current.
/// The lock-down progress bar under a seat's playfield: width tracks progress
/// (`1.0 - lock_timer_fraction` — the engine reports the fraction REMAINING),
/// visible only while the seat's active piece is grounded.
#[derive(Component)]
pub struct SeatTimerBar {
    pub seat: usize,
}

fn update_seat_timer_bars(
    seats: Query<(&Seat, &SeatSnapshot)>,
    mut bars: Query<(&SeatTimerBar, &mut Sprite)>,
) {
    for (seat, snapshot) in &seats {
        for (bar, mut sprite) in &mut bars {
            if bar.seat != seat.index {
                continue;
            }
            let progress = lock_bar_progress(&snapshot.0);
            let width = SessionLayout::BLOCK * SessionLayout::BOARD_W as f32 * progress;
            if let Some(size) = sprite.custom_size {
                sprite.custom_size = Some(Vec2::new(width, size.y));
            }
        }
    }
}

/// Highest occupied row of a snapshot's stack, or `-1` for an empty board.
fn stack_peak_row(snapshot: &crate::engine::EngineSnapshot) -> isize {
    snapshot
        .board_cells
        .iter()
        .map(|cell| cell.y)
        .max()
        .unwrap_or(-1)
}

/// First board row of the danger zone, with the ramp spanning it and the
/// rows above (16..=19; buffer-zone cells clamp to full danger).
const DANGER_ZONE_ROW: isize = 16;
const DANGER_ZONE_SPAN: f32 = 4.0;

/// How deep a stack sits in the danger zone, 0..=1 — the ONE definition of
/// "the stack is dangerously high", consumed by the frame-warming pass here
/// and the ambient background's calm state (`features::ambient_wave`).
pub(crate) fn danger_level(snapshot: &crate::engine::EngineSnapshot) -> f32 {
    let rows_in = (stack_peak_row(snapshot) - DANGER_ZONE_ROW + 1) as f32;
    (rows_in / DANGER_ZONE_SPAN).clamp(0.0, 1.0)
}

/// Danger state: warm a seat's field frame toward `ATTACK` as its stack
/// climbs the top four visible rows. The signal lives entirely in the frame —
/// the field background never changes during play, under any circumstance.
fn tint_danger_frames(
    seats: Query<(&Seat, &SeatSnapshot)>,
    mut frames: Query<(&BoardFrame, &mut Sprite)>,
) {
    use bevy::color::Mix;
    for (seat, snapshot) in &seats {
        let danger = danger_level(&snapshot.0);
        let color =
            crate::ui::widgets::theme::FRAME.mix(&crate::ui::widgets::theme::ATTACK, 0.55 * danger);
        for (frame, mut sprite) in &mut frames {
            if frame.seat == seat.index && sprite.color != color {
                sprite.color = color;
            }
        }
    }
}

/// Lock-bar fill for a snapshot: only a GROUNDED piece shows progress. The
/// engine reports `lock_timer_fraction` as the fraction REMAINING, and `0.0`
/// also means "timer not running" (a falling piece) — so without the `landed`
/// gate every falling piece reads as a permanently full bar.
fn lock_bar_progress(snapshot: &crate::engine::EngineSnapshot) -> f32 {
    match snapshot.active.as_ref() {
        Some(active) if active.landed => 1.0 - active.lock_timer_fraction,
        _ => 0.0,
    }
}

fn update_atk_texts(
    config: Res<super::SessionConfig>,
    clock: Res<super::MatchClock>,
    seats: Query<(&Seat, &SeatStats, &SeatSnapshot)>,
    mut texts: Query<(&SeatAtkText, &mut Text2d)>,
) {
    for (seat, stats, snapshot) in &seats {
        for (atk, mut text) in &mut texts {
            if atk.seat != seat.index {
                continue;
            }
            let line = match config.mode {
                // Versus: the pressure scoreboard.
                super::SessionMode::Versus => format!("ATK {}", stats.attack_sent),
                // Solo: the run line — score, lines, level, and the variant
                // clock (Sprint counts up; Ultra counts down to its limit).
                super::SessionMode::Solo { variant } => {
                    let snap = &snapshot.0;
                    let shown = match variant.def().end_condition {
                        crate::variant::EndCondition::TimeLimit(limit) => {
                            (limit - clock.0).max(0.0)
                        }
                        _ => clock.0,
                    };
                    format!(
                        "SCORE {}   LINES {}   LVL {}   {}:{:04.1}",
                        snap.score,
                        snap.lines,
                        snap.level,
                        (shown / 60.0) as u32,
                        shown % 60.0
                    )
                }
            };
            if text.0 != line {
                text.0 = line;
            }
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn the_garbage_meter_floats_with_equal_air_on_both_sides() {
        let block = SessionLayout::BLOCK;
        let air = SessionLayout::METER_AIR_CELLS * block;
        // Seat 0's inner edge, left to right: frame → air → meter → air →
        // preview column (the same x math as spawn_session_scene).
        let board_right = SessionLayout::BOARD_W as f32 * block;
        let meter_left = board_right + air;
        let column_x =
            (SessionLayout::BOARD_W as f32 + SessionLayout::column_gap_cells(true)) * block;
        assert_eq!(meter_left - board_right, air);
        assert_eq!(column_x - (meter_left + METER_WIDTH), air);
        // Seat 1 mirrors: hold column → air → meter → air → frame.
        let meter_left = -air - METER_WIDTH;
        let hold_x = -SessionLayout::column_gap_cells(true) * block;
        assert_eq!(meter_left - hold_x, air);
        assert_eq!(0.0 - (meter_left + METER_WIDTH), air);
        // Without a meter (solo, outer flanks) the column keeps its snug
        // half-cell offset.
        assert_eq!(SessionLayout::column_gap_cells(false), 0.5);
    }
}

#[cfg(test)]
mod timer_bar_tests {
    use super::lock_bar_progress;
    use crate::engine::{Engine, EngineConfig, InputFrame};

    #[test]
    fn a_falling_piece_shows_no_lock_progress() {
        // The regression: lock_timer_fraction is 0.0 for an INACTIVE timer,
        // which without the landed gate read as a full bar on every falling
        // piece (user-reported: "the bar shows up by default").
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn; piece is high and falling
        let snapshot = engine.snapshot();
        assert!(!snapshot.active.as_ref().unwrap().landed);
        assert_eq!(lock_bar_progress(&snapshot), 0.0);
    }

    #[test]
    fn a_grounded_piece_fills_the_bar_as_the_timer_drains() {
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default());
        // Hard-drop-free grounding: step gravity until the piece lands.
        for _ in 0..4000 {
            engine.step(InputFrame {
                dt_seconds: 1.0 / 60.0,
                soft_drop: true,
                ..Default::default()
            });
            let snap = engine.snapshot();
            if snap.active.as_ref().is_some_and(|a| a.landed) {
                // Just landed: full timer remaining ⇒ bar starts (near) empty.
                assert!(lock_bar_progress(&snap) < 0.1);
                // Drain half the lock-down; the bar fills accordingly.
                engine.step(InputFrame {
                    dt_seconds: 0.25,
                    ..Default::default()
                });
                let drained = engine.snapshot();
                if drained.active.as_ref().is_some_and(|a| a.landed) {
                    assert!(
                        lock_bar_progress(&drained) > 0.3,
                        "a draining lock timer must fill the bar"
                    );
                }
                return;
            }
        }
        panic!("the piece never landed");
    }

    #[test]
    fn no_active_piece_shows_no_lock_progress() {
        let engine = Engine::new(EngineConfig::default(), 7);
        let snapshot = engine.snapshot(); // pre-spawn: no active piece
        assert!(snapshot.active.is_none());
        assert_eq!(lock_bar_progress(&snapshot), 0.0);
    }
}
