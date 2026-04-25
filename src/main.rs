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
