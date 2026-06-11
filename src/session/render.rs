//! Versus rendering: two board groups, everything parented per seat.
//!
//! Each seat gets a **board root** entity at its world-space origin; the
//! background grid, the mino layers (locked / falling / ghost), the hold and
//! preview columns, the garbage meter, and the seat texts are all children of
//! (or anchored to) that root — position is one transform, despawn is one
//! subtree, and a future mirrored layout is a per-root parameter. One camera
//! frames both boards with `ScalingMode::AutoMin`, so native window resizes
//! and the web canvas keep the whole match visible. The camera carries the
//! [`GameplayCamera`] tag, so the bloom/CRT stack applies exactly as in
//! single-player (the screen-shake mover is `Playing`-gated and never touches
//! it).
//!
//! Reconcilers mirror the single-player pattern — diff the cached snapshot
//! slice, despawn-and-respawn only what changed — but query per seat instead
//! of per global marker. Garbage cells (`SnapshotCell::garbage`) paint a
//! neutral gray: telling your own stack from their attack at a glance is the
//! point of having a versus renderer at all.

use bevy::camera::ScalingMode;
use bevy::prelude::*;
use bevy::sprite::Anchor;

use crate::assets::GameAssets;
use crate::engine::{Piece, PieceType, SnapshotCell};
use crate::level::common::{mino_render_color, to_translation, GameplayCamera};
use crate::GameState;

use super::{Participant, Seat, SeatSnapshot, SeatStats, SessionConfig};

/// World-space layout of the two-board scene, in cells and pixels. One home
/// for every magic number the renderer and overlays share.
pub struct SessionLayout;

impl SessionLayout {
    pub const BLOCK: f32 = 32.0;
    pub const BOARD_W: usize = 10;
    pub const BOARD_H: usize = 20;
    /// Cells between the two boards — wide enough for seat 0's preview column
    /// and seat 1's hold column to meet without touching.
    pub const GUTTER_CELLS: f32 = 10.0;
    /// Avatar scale for hold/preview minos (matches the single-player feel).
    pub const PREVIEW_SCALE: f32 = 0.8;

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
    /// board, the outer hold/preview columns, the texts above and below.
    pub fn scene_min(seat_count: usize) -> (f32, f32) {
        let width_cells = match seat_count {
            // hold column + board + preview column + breathing room.
            0 | 1 => 20.0,
            // two board groups + the gutter between them.
            _ => 40.0,
        };
        (width_cells * Self::BLOCK, 25.0 * Self::BLOCK)
    }
}

/// Neutral gray for garbage cells — full alpha (the half-alpha gray is the
/// ghost's), deliberately desaturated against the seven piece colours.
pub fn garbage_color() -> Color {
    Color::srgb_u8(112, 116, 122)
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

/// The garbage meter root for a seat; children are the batch segments.
#[derive(Component)]
pub struct SeatMeter {
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

pub struct VersusRenderPlugin;

impl Plugin for VersusRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Session), setup_scene)
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
                )
                    .run_if(in_state(GameState::Session)),
            );
    }
}

/// One mino-sized sprite at board-relative cell `(x, y)` — the versus block
/// primitive (the shared `spawn_free_block` couples colour to `PieceType`,
/// which garbage cells deliberately do not have).
fn block_sprite(
    assets: &GameAssets,
    block_size: f32,
    x: isize,
    y: isize,
    color: Color,
    z: f32,
) -> impl Bundle {
    let mut sprite = Sprite::from_image(assets.block_texture.clone());
    sprite.custom_size = Some(Vec2::splat(block_size));
    sprite.color = color;
    let mut transform = Transform::from_translation(to_translation(x, y, block_size));
    transform.translation.z = z;
    (sprite, transform, Anchor::BOTTOM_LEFT)
}

/// The colour of a snapshot cell: garbage gray, or the piece's render colour.
fn cell_color(cell: &SnapshotCell) -> Color {
    if cell.garbage {
        garbage_color()
    } else {
        mino_render_color(cell.piece_type)
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

        // Background grid (decorative, drawn once).
        let mut grid = Vec::new();
        for x in 0..SessionLayout::BOARD_W as isize {
            for y in 0..SessionLayout::BOARD_H as isize {
                let mut sprite = Sprite::from_color(Color::srgb(0.1, 0.1, 0.1), Vec2::splat(block));
                sprite.custom_size = Some(Vec2::splat(block));
                let mut transform = Transform::from_translation(to_translation(x, y, block));
                transform.translation.z = -1.0;
                grid.push(
                    commands
                        .spawn((sprite, transform, Anchor::BOTTOM_LEFT))
                        .id(),
                );
            }
        }
        commands.entity(root).add_children(&grid);

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
        // seat 0's on its right, seat 1's on its left. Segments stack upward
        // from the board floor, one per pending batch.
        let meter_x = if seat == 0 {
            SessionLayout::BOARD_W as f32 * block + 0.35 * block
        } else {
            -0.7 * block
        };
        let meter = commands
            .spawn((
                SeatMeter { seat },
                Transform::from_translation(Vec3::new(meter_x, 0.0, 0.5)),
                Visibility::default(),
            ))
            .id();
        commands.entity(root).add_child(meter);

        // Hold column (top-left of the board) and preview column (top-right) —
        // the single-player arrangement, duplicated per seat.
        let hold = commands
            .spawn((
                SeatHoldView { seat },
                Transform::from_translation(Vec3::new(
                    -0.5 * block,
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
                    (SessionLayout::BOARD_W as f32 + 0.5) * block,
                    SessionLayout::BOARD_H as f32 * block,
                    0.0,
                )),
                Visibility::default(),
            ))
            .id();
        commands.entity(root).add_children(&[hold, preview]);

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
                    font_size: 28.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Anchor::BOTTOM_CENTER,
                Transform::from_translation(Vec3::new(
                    SessionLayout::BOARD_W as f32 * block / 2.0,
                    (SessionLayout::BOARD_H as f32 + 0.6) * block,
                    0.0,
                )),
            ))
            .id();

        // Cumulative attack under the board.
        let atk_id = commands
            .spawn((
                SeatAtkText { seat },
                Text2d::new("ATK 0"),
                TextFont {
                    font: assets.font.clone(),
                    font_size: 20.0,
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.55, 0.55)),
                Anchor::TOP_CENTER,
                Transform::from_translation(Vec3::new(
                    SessionLayout::BOARD_W as f32 * block / 2.0,
                    -0.6 * block,
                    0.0,
                )),
            ))
            .id();
        commands.entity(root).add_children(&[label_id, atk_id]);
    }

    // One camera, both boards always in frame. `GameplayCamera` opts into the
    // bloom/CRT stack; the shake mover is gated on `Playing` and never runs here.
    commands.spawn((
        Camera2d,
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
    assets: Res<GameAssets>,
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
        let ids: Vec<Entity> = cells
            .iter()
            .map(|cell| {
                commands
                    .spawn(block_sprite(
                        &assets,
                        SessionLayout::BLOCK,
                        cell.x,
                        cell.y,
                        cell_color(cell),
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
    assets: Res<GameAssets>,
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
        let ids: Vec<Entity> = active
            .cells
            .iter()
            .map(|cell| {
                commands
                    .spawn(block_sprite(
                        &assets,
                        SessionLayout::BLOCK,
                        cell.x,
                        cell.y,
                        cell_color(cell),
                        0.0,
                    ))
                    .id()
            })
            .collect();
        commands.entity(layer).add_children(&ids);
    }
}

/// Rebuild each seat's ghost every frame (hidden when grounded or disabled).
fn reconcile_ghost_pieces(
    mut commands: Commands,
    assets: Res<GameAssets>,
    settings: Res<crate::settings::GameSettings>,
    seats: Query<(&Seat, &SeatSnapshot)>,
    layers: Query<(Entity, &VsLayer, &LayerSeat)>,
) {
    for (seat, snapshot) in &seats {
        let Some(layer) = layer_for(&layers, seat.index, VsLayer::Ghost) else {
            continue;
        };
        commands.entity(layer).despawn_related::<Children>();
        if !settings.ghost_enabled {
            continue;
        }
        let landed = snapshot
            .0
            .active
            .as_ref()
            .is_none_or(|active| active.landed);
        if landed {
            continue;
        }
        let ids: Vec<Entity> = snapshot
            .0
            .ghost_cells
            .iter()
            .map(|cell| {
                commands
                    .spawn(block_sprite(
                        &assets,
                        SessionLayout::BLOCK,
                        cell.x,
                        cell.y,
                        Color::srgb(0.5, 0.5, 0.5).with_alpha(0.5),
                        -0.1,
                    ))
                    .id()
            })
            .collect();
        commands.entity(layer).add_children(&ids);
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
                Color::srgb_u8(214, 48, 49),
                Vec2::new(0.35 * SessionLayout::BLOCK, height.max(NOTCH)),
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
    assets: &GameAssets,
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
    let ids: Vec<Entity> = piece
        .avatar_cells()
        .iter()
        .map(|&(x, y)| {
            commands
                .spawn(block_sprite(
                    assets,
                    block,
                    x,
                    y,
                    mino_render_color(piece_type),
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
    assets: Res<GameAssets>,
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
            spawn_avatar(&mut commands, &assets, view, piece_type, 0.0, true);
        }
        cache[index] = Some(hold);
    }
}

/// Rebuild a seat's next-queue column when the visible queue changes.
fn reconcile_preview_views(
    mut commands: Commands,
    assets: Res<GameAssets>,
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
            let height = spawn_avatar(&mut commands, &assets, view, piece_type, y_top, false);
            y_top -= height + gap;
        }
        state.cache = Some(queue.clone());
    }
}

/// Keep each seat's cumulative-attack readout current.
fn update_atk_texts(
    seats: Query<(&Seat, &SeatStats), Changed<SeatStats>>,
    mut texts: Query<(&SeatAtkText, &mut Text2d)>,
) {
    for (seat, stats) in &seats {
        for (atk, mut text) in &mut texts {
            if atk.seat == seat.index {
                text.0 = format!("ATK {}", stats.attack_sent);
            }
        }
    }
}
