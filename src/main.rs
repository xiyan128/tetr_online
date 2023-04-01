use bevy::prelude::*;
use tetr_online::GamePlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        .add_plugins(DefaultPlugins
            .set(ImagePlugin::default_nearest())
            .set(WindowPlugin {
                primary_window: Some(Window {
                    fit_canvas_to_parent: true,
                    canvas: Some("#bevy".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }))
        .add_plugin(GamePlugin)
        .run();
}
