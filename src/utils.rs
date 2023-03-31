use bevy::prelude::*;

pub fn continue_to_state<T: States>(state: T) -> impl FnMut(ResMut<NextState<T>>) {
    move |mut next_state: ResMut<NextState<T>>| {
        next_state.set(state.clone());
    }
}
