use crate::level::common::{GameField, LevelConfig};
use crate::level::engine_bridge::LatestSnapshot;
use bevy::ecs::error::Result;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct LockingTimerBar;

pub fn spawn_locking_timer_bar(
    mut commands: Commands,
    config: Res<LevelConfig>,
    // No field ⇒ `Single` skips the system; nothing to parent the bar to.
    field: Single<Entity, With<GameField>>,
) {
    let field = *field;
    let bar_height = config.block_size * 0.2;

    // spawn the timer bar
    let timer_bar_entity = commands
        .spawn((
            Sprite {
                custom_size: Some(Vec2::new(
                    config.block_size * config.board_width as f32,
                    bar_height,
                )),
                color: Color::srgb(0.5, 0.5, 0.5),
                ..Default::default()
            },
            Anchor::BOTTOM_LEFT,
            Transform::from_translation(Vec3::new(0., -bar_height, 1.)),
        ))
        .insert(LockingTimerBar)
        .id();

    commands.entity(field).add_child(timer_bar_entity);
}

/// Width of the lock-down bar tracks lock-down *progress*. The engine's
/// `lock_timer_fraction` is the fraction of lock-down time *remaining*
/// (1.0 at landing → 0.0 at lock), so progress (the bar that grows toward lock,
/// matching the pre-migration visual) is `1.0 - lock_timer_fraction`.
pub fn update_locking_timer_bar(
    snapshot: Res<LatestSnapshot>,
    mut bar: Single<&mut Sprite, With<LockingTimerBar>>,
    config: Res<LevelConfig>,
) -> Result {
    let remaining = snapshot
        .0
        .active
        .as_ref()
        .map(|active| active.lock_timer_fraction)
        .unwrap_or(1.0);
    let progress = 1.0 - remaining;
    let width = config.block_size * config.board_width as f32 * progress;

    // The bar is always spawned with an explicit `custom_size`; if it is somehow
    // absent there's nothing to resize, so bail out cleanly rather than panic.
    let Some(size) = bar.custom_size else {
        return Ok(());
    };
    bar.custom_size = Some(Vec2::new(width, size.y));
    Ok(())
}

pub fn despawn_locking_timer_bar(
    // No bar ⇒ `Single` skips the system; nothing to despawn.
    bar: Single<Entity, With<LockingTimerBar>>,
    mut commands: Commands,
) {
    commands.entity(*bar).despawn();
}
