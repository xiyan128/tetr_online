use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

#[derive(AssetCollection, Resource)]
pub struct GameAssets {
    #[asset(path = "textures/block_tile.png")]
    pub block_texture: Handle<Image>,

    #[asset(path = "sounds/drop.ogg")]
    pub hard_drop_sound: Handle<AudioSource>,

    #[asset(path = "sounds/drop.ogg")]
    pub placed_sound: Handle<AudioSource>,

    #[asset(path = "sounds/clear_1.ogg")]
    pub line_clear_1: Handle<AudioSource>,

    #[asset(path = "sounds/clear_2.ogg")]
    pub line_clear_2: Handle<AudioSource>,

    #[asset(path = "sounds/clear_3.ogg")]
    pub line_clear_3: Handle<AudioSource>,

    #[asset(path = "sounds/clear_4.ogg")]
    pub line_clear_4: Handle<AudioSource>,

    #[asset(path = "sounds/lock.ogg")]
    pub locked_sound: Handle<AudioSource>,

    #[asset(path = "sounds/hold.ogg")]
    pub hold_sound: Handle<AudioSource>,

    #[asset(path = "sounds/rotate.ogg")]
    pub rotation_sound: Handle<AudioSource>,

    #[asset(path = "fonts/dogicabold.ttf")]
    pub font: Handle<Font>,
}
