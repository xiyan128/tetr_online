use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

#[derive(AssetCollection, Resource)]
pub struct GameAssets {
    #[asset(path = "textures/block_tile.png")]
    pub block_texture: Handle<Image>,

    // sounds
    // #[asset(path = "sounds/rotate.wav")]
    // pub rotate_sound: Handle<AudioSource>,
    //
    // #[asset(path = "sounds/spin.wav")]
    // pub lock_sound: Handle<AudioSource>,
    //
    // #[asset(path = "sounds/line_clear.wav")]
    // pub line_clear_sound: Handle<AudioSource>,
    #[asset(path = "sounds/drop.ogg")]
    pub soft_drop_sound: Handle<AudioSource>,

    #[asset(path = "sounds/drop.ogg")]
    pub hard_drop_sound: Handle<AudioSource>,

    #[asset(path = "fonts/dogicabold.ttf")]
    pub font: Handle<Font>,
    // #[asset(path = "sounds/hold.wav")]
    // pub hold_sound: Handle<AudioSource>,
}
