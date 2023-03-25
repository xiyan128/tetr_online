use bevy::prelude::{Input, KeyCode, Res, States};

pub enum GameControl {
    Up,
    Down,
    Left,
    Right,
    RotateClockwise,
    RotateCounterClockwise,
}

impl GameControl {
    pub fn pressed(&self, keyboard_input: &Res<Input<KeyCode>>) -> bool {
        match self {
            GameControl::Up => {
                keyboard_input.pressed(KeyCode::W) || keyboard_input.pressed(KeyCode::Up)
            }
            GameControl::Down => {
                keyboard_input.pressed(KeyCode::S) || keyboard_input.pressed(KeyCode::Down)
            }
            GameControl::Left => {
                keyboard_input.pressed(KeyCode::A) || keyboard_input.pressed(KeyCode::Left)
            }
            GameControl::Right => {
                keyboard_input.pressed(KeyCode::D) || keyboard_input.pressed(KeyCode::Right)
            }
            GameControl::RotateClockwise => keyboard_input.just_pressed(KeyCode::X),
            GameControl::RotateCounterClockwise => keyboard_input.just_pressed(KeyCode::Z),
        }
    }
}

pub fn get_action(control: GameControl, input: &Res<Input<KeyCode>>) -> i32 {
    control.pressed(input) as i32
}