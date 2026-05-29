//! Binary entry point: builds the Bevy `App` and runs [`GamePlugin`].
//!
//! Configures `DefaultPlugins` for both native and web targets — nearest-
//! neighbour image sampling for crisp blocks, no asset meta lookups (so the
//! browser build doesn't 404 on `.meta` files), and a window bound to the
//! `#bevy` canvas that resizes with its parent. All game logic lives in
//! [`GamePlugin`] from the library crate.

use bevy::asset::AssetMetaCheck;
use bevy::prelude::*;
use tetr_online::GamePlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.2, 0.2, 0.2)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        fit_canvas_to_parent: true,
                        canvas: Some("#bevy".to_owned()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
        )
        .add_plugins(GamePlugin)
        .run();
}
