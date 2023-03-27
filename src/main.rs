use bevy::prelude::*;
use tetr_online::GamePlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::DARK_GRAY))
        // .add_plugins(DefaultPlugins.set(WindowPlugin {
        //     primary_window: Some(Window {
        //         title: "Bevy game".to_string(), // ToDo
        //         resolution: (800., 600.).into(),
        //         canvas: Some("#bevy".to_owned()),
        //         ..default()
        //     }),
        //     ..default()
        // }))
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .add_plugin(GamePlugin)
        .run();
}
