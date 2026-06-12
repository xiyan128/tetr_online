//! Binary entry point: builds the Bevy `App` and runs [`GamePlugin`].
//!
//! Configures `DefaultPlugins` for both native and web targets — nearest-
//! neighbour image sampling for crisp blocks, no asset meta lookups (so the
//! browser build doesn't 404 on `.meta` files), and a window bound to the
//! `#bevy` canvas that resizes with its parent. On macOS the title bar is
//! dissolved into the game (transparent, no title text, content drawn behind
//! it) for a seamless desktop frame. All game logic lives in [`GamePlugin`]
//! from the library crate.

use bevy::asset::AssetMetaCheck;
use bevy::prelude::*;
use tetr_online::GamePlugin;

fn main() {
    // On the web, route panics to the browser console with a readable stack.
    // No-op on native (the hook crate is a wasm-only dependency).
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    App::new()
        // Kissaten ground: warm charcoal (#2E2B28) — the field and panel bg.
        .insert_resource(ClearColor(Color::srgb(0.1804, 0.1686, 0.1569)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..Default::default()
                })
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "TETR ONLINE".to_owned(),
                        fit_canvas_to_parent: true,
                        canvas: Some("#bevy".to_owned()),
                        // macOS: dissolve the title bar into the game. The
                        // charcoal field renders edge-to-edge behind a
                        // transparent bar with no title text; only the
                        // traffic-light buttons float over the top-left,
                        // kept so the window stays movable and closable.
                        // These fields are no-ops on web and Linux.
                        fullsize_content_view: true,
                        titlebar_transparent: true,
                        titlebar_show_title: false,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
        )
        .add_plugins(GamePlugin)
        .run();
}
