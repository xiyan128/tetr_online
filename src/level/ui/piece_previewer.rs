//! Next-queue and hold-slot previews.
//!
//! Renders the upcoming pieces from `snapshot.next_queue` and the held piece
//! from `snapshot.hold` as small avatar boards beside the playfield. Both views
//! cache the last-rendered state and only rebuild their sprites when the queue
//! or hold contents change.

use crate::assets::GameAssets;
use crate::engine::{Piece, PieceType};
use crate::level::common::spawn_free_block;
use crate::level::common::{to_translation, BlockKind, LevelConfig};
use crate::level::engine_bridge::LatestSnapshot;
use crate::level::ui::calc_ui_offset;
use crate::InGameplay;
use bevy::ecs::error::Result;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PiecePreviewer;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct HoldViewer;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PreviewHolder {
    index: usize,
}

pub fn spawn_piece_previewer(mut commands: Commands, config: Res<LevelConfig>) {
    info!("spawning piece previewer");
    let offset = Vec3::new(calc_ui_offset(&config), 0., 0.);

    let piece_previewer_entity = commands
        .spawn(PiecePreviewer)
        .insert(Transform::from_translation(
            to_translation(
                config.board_width as isize,
                (config.board_height) as isize,
                config.block_size,
            ) + offset,
        ))
        .insert(DespawnOnExit(InGameplay))
        .id();

    for i in 0..config.preview_count {
        let box_entity = commands
            .spawn((Transform::default(), PreviewHolder { index: i }))
            .id();

        commands
            .entity(piece_previewer_entity)
            .add_child(box_entity);
    }
}

pub fn spawn_hold_viewer(mut commands: Commands, config: Res<LevelConfig>) {
    let offset = Vec3::new(-calc_ui_offset(&config), 0., 0.);

    commands.spawn((
        HoldViewer,
        Transform::from_translation(
            to_translation(0, (config.board_height) as isize, config.block_size) + offset,
        ),
        DespawnOnExit(InGameplay),
        children![(Transform::default(), PreviewHolder { index: 0 })],
    ));
}

/// Render the next-piece previews from `snapshot.next_queue`. Cached against the
/// last-rendered queue so sprites are only rebuilt when the queue changes.
pub fn update_piece_previewer(
    children: Single<&Children, With<PiecePreviewer>>,
    mut preview_holder_query: Query<(&mut PreviewHolder, &mut Transform)>,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
    mut commands: Commands,
    mut last_queue: Local<Option<Vec<PieceType>>>,
) -> Result {
    let queue = &snapshot.0.next_queue;
    if last_queue.as_ref() == Some(queue) {
        return Ok(());
    }

    let mut holders_height = 0.;
    for child in children.iter() {
        // clear the preview holder
        commands.entity(child).despawn_related::<Children>();

        // Each child of the previewer is a `PreviewHolder`; propagate rather than
        // panic if that invariant is ever violated.
        let (preview_holder, mut holder_transform) = preview_holder_query.get_mut(child)?;

        let idx = preview_holder.index;
        let Some(&piece_type) = queue.get(idx) else {
            continue;
        };

        let (preview_board_size, piece_entity) =
            spawn_holder_piece(&config, &game_assets, &mut commands, piece_type);

        commands.entity(child).add_child(piece_entity);

        // stack the holders
        holders_height += preview_board_size.y + calc_ui_offset(&config);
        holder_transform.translation.y = -holders_height;
    }

    *last_queue = Some(queue.clone());
    Ok(())
}

/// Render the hold piece from `snapshot.hold`. Cached against the last-rendered
/// hold so sprites are only rebuilt when it changes.
pub fn update_hold_viewer(
    children: Single<&Children, With<HoldViewer>>,
    mut preview_holder_query: Query<(&mut PreviewHolder, &mut Transform)>,
    snapshot: Res<LatestSnapshot>,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
    mut commands: Commands,
    mut last_hold: Local<Option<Option<PieceType>>>,
) -> Result {
    let hold = snapshot.0.hold;
    if *last_hold == Some(hold) {
        return Ok(());
    }

    for child in children.iter() {
        // clear the preview holder
        commands.entity(child).despawn_related::<Children>();

        if let Some(piece_type) = hold {
            // The hold viewer's child is a `PreviewHolder`; propagate rather than
            // panic if that invariant is ever violated.
            let (_, mut holder_transform) = preview_holder_query.get_mut(child)?;

            let (preview_board_size, piece_entity) =
                spawn_holder_piece(&config, &game_assets, &mut commands, piece_type);

            commands.entity(child).add_child(piece_entity);

            // move down and left by board size
            holder_transform.translation = -preview_board_size.extend(0.);
        }
    }

    *last_hold = Some(hold);
    Ok(())
}

fn spawn_holder_piece(
    config: &Res<LevelConfig>,
    game_assets: &Res<GameAssets>,
    commands: &mut Commands,
    piece_type: PieceType,
) -> (Vec2, Entity) {
    let piece = Piece::from(piece_type);

    let avatar_board = piece.avatar_board();

    let preview_block_size = config.block_size * config.preview_scale;
    let preview_config = LevelConfig {
        block_size: preview_block_size,
        ..Default::default()
    };

    let block_ids: Vec<Entity> = avatar_board
        .cells()
        .iter()
        .map(|&cell| {
            spawn_free_block(
                commands,
                &preview_config,
                game_assets,
                cell,
                BlockKind::Preview,
            )
        })
        .collect();

    let scale = match piece_type {
        PieceType::I => 0.95,
        PieceType::O => 1.05,
        _ => 1.0,
    }; // slightly scale the preview to fit the board

    let preview_board_size = Vec2::new(avatar_board.width() as f32, avatar_board.height() as f32)
        * preview_block_size
        * scale;

    let piece_entity = commands
        .spawn((
            Sprite {
                custom_size: Some(preview_board_size),
                color: Color::srgba(0.0, 0.0, 0.0, 0.0),
                ..Default::default()
            },
            Anchor::BOTTOM_LEFT,
            Transform::from_scale(Vec3::splat(scale)),
        ))
        .id();

    for block_id in block_ids {
        commands.entity(piece_entity).add_child(block_id);
    }
    (preview_board_size, piece_entity)
}
