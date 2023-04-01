use crate::assets::GameAssets;
use crate::core::{Board, Piece, PieceGenerator, PieceType};
use crate::level::common::spawn_free_block;
use crate::level::common::{to_translation, FallingBlock, LevelCleanup, LevelConfig, PieceHolder};
use crate::level::ui::calc_ui_offset;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component)]
pub struct PiecePreviewer;

#[derive(Component)]
pub struct HoldViewer;

#[derive(Component)]
pub struct PreviewHolder {
    index: usize,
}

pub fn spawn_piece_previewer(mut commands: Commands, config: Res<LevelConfig>) {
    info!("spawning piece previewer");
    let offset = Vec3::new(calc_ui_offset(&config), 0., 0.);

    let piece_previewer_entity = commands
        .spawn(PiecePreviewer)
        .insert(SpatialBundle {
            transform: Transform::from_translation(
                to_translation(
                    config.board_width as isize,
                    (config.board_height) as isize,
                    config.block_size,
                ) + offset,
            ),
            ..Default::default()
        })
        .insert(LevelCleanup)
        .id();

    for i in 0..config.preview_count {
        let box_entity = commands
            .spawn((SpatialBundle::default(), PreviewHolder { index: i }))
            .id();

        commands
            .entity(piece_previewer_entity)
            .add_child(box_entity);
    }
}

pub fn spawn_hold_viewer(mut commands: Commands, config: Res<LevelConfig>) {
    let offset = Vec3::new(-calc_ui_offset(&config), 0., 0.);

    let hold_viewer = commands
        .spawn(HoldViewer)
        .insert(SpatialBundle {
            transform: Transform::from_translation(
                to_translation(0, (config.board_height) as isize, config.block_size) + offset,
            ),
            ..Default::default()
        })
        .insert(LevelCleanup)
        .id();

    let box_entity = commands
        .spawn((SpatialBundle::default(), PreviewHolder { index: 0 }))
        .id();

    commands.entity(hold_viewer).add_child(box_entity);
}

pub fn update_piece_previewer(
    children_query: Query<&Children, With<PiecePreviewer>>,
    mut preview_holder_query: Query<(&mut PreviewHolder, &mut Transform)>,
    mut generator_query: Query<&mut PieceGenerator, Changed<PieceGenerator>>,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
    mut commands: Commands,
) {
    if generator_query.is_empty() {
        return;
    }
    info!("updating piece previewer");

    let mut generator = generator_query.single_mut();

    let children = children_query.single();

    let bag_preview = generator.preview();

    let mut holders_height = 0.;
    for child in children.iter() {
        // clear the preview holder
        commands.entity(*child).despawn_descendants();

        let (preview_holder, mut holder_transform) = preview_holder_query.get_mut(*child).unwrap();

        let idx = preview_holder.index;

        let piece_type = bag_preview[idx];

        let (preview_board_size, piece_entity) =
            spawn_holder_piece(&config, &game_assets, &mut commands, piece_type);

        commands.entity(*child).add_child(piece_entity);

        // stack the holders
        holders_height += preview_board_size.y + calc_ui_offset(&config);
        holder_transform.translation.y = -holders_height;
    }
}

pub fn update_hold_viewer(
    children_query: Query<&Children, With<HoldViewer>>,
    mut preview_holder_query: Query<(&mut PreviewHolder, &mut Transform)>,
    mut holder_query: Query<&PieceHolder>,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
    mut commands: Commands,
) {
    if children_query.is_empty() || holder_query.is_empty() {
        return;
    }

    let piece_holder = holder_query.single_mut();

    let children = children_query.single();

    for child in children.iter() {
        // clear the preview holder
        commands.entity(*child).despawn_descendants();

        if let Some(piece_type) = piece_holder.piece.clone() {
            let (preview_holder, mut holder_transform) =
                preview_holder_query.get_mut(*child).unwrap();

            let (preview_board_size, piece_entity) = spawn_holder_piece(
                &config,
                &game_assets,
                &mut commands,
                piece_type.piece_type(),
            );

            commands.entity(*child).add_child(piece_entity);

            // move down and left by board size
            holder_transform.translation = -preview_board_size.extend(0.);
        }
    }
}

fn spawn_holder_piece(
    config: &Res<LevelConfig>,
    game_assets: &Res<GameAssets>,
    mut commands: &mut Commands,
    piece_type: PieceType,
) -> (Vec2, Entity) {
    let piece = Piece::from(piece_type);

    let avatar_board = piece.avatar_board();

    let preview_block_size = config.block_size * config.preview_scale;

    let block_ids: Vec<Entity> = avatar_board
        .cells()
        .iter()
        .map(|&cell| {
            spawn_free_block(
                &mut commands,
                &LevelConfig {
                    block_size: preview_block_size,
                    ..Default::default()
                },
                &game_assets,
                cell,
                FallingBlock,
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
        .spawn(piece)
        .insert(SpriteBundle {
            sprite: Sprite {
                custom_size: Some(preview_board_size),
                color: Color::NONE,
                anchor: Anchor::BottomLeft,
                ..Default::default()
            },
            transform: Transform::from_scale(Vec3::splat(scale)),
            ..Default::default()
        })
        .id();

    for block_id in block_ids {
        commands.entity(piece_entity).add_child(block_id);
    }
    (preview_board_size, piece_entity)
}
