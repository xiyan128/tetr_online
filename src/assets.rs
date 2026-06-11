//! The game's loaded asset handles.
//!
//! [`GameAssets`] is a `bevy_asset_loader` collection populated during the
//! loading state: the block texture, the UI font, and the sound-effect sources.
//! Systems take it as a resource rather than loading assets ad hoc.

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

    /// The display voice (Kissaten): Dogica, a pixel font on an 8 px grid —
    /// titles, stat numerals, button labels, callouts. Crisp only at native
    /// multiples (16 / 24 / 32 / 96), so every size in `theme` is one.
    #[asset(path = "fonts/dogicabold.ttf")]
    pub font: Handle<Font>,

    /// The working voice: Departure Mono (11 px grid) — body copy, hints,
    /// menus, tables. Sentence case, 14 px body / 12 px micro.
    #[asset(path = "fonts/DepartureMono-Regular.otf")]
    pub font_body: Handle<Font>,
}
