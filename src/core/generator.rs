use crate::core::pieces::PieceType;
use rand::seq::SliceRandom;

use bevy::prelude::Component;
use std::iter::Iterator;

#[derive(Component)]
pub struct PieceGenerator {
    bag: Vec<PieceType>,
}

impl PieceGenerator {
    pub fn new() -> Self {
        let mut bag = PieceType::all();
        let mut rng = rand::thread_rng();
        bag.shuffle(&mut rng);

        Self { bag }
    }

    fn refill_bag(&mut self) {
        let mut next_bag = PieceType::all();
        let mut rng = rand::thread_rng();
        next_bag.shuffle(&mut rng);

        self.bag = [&next_bag[..], &self.bag[..]].concat();
    }

    pub(crate) fn preview(&mut self) -> Vec<PieceType> {
        if self.bag.len() < PieceType::LEN {
            self.refill_bag();
        }
        self.bag[self.bag.len() - PieceType::LEN..]
            .iter()
            .rev()
            .copied()
            .collect()
    }
}

impl Default for PieceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for PieceGenerator {
    type Item = PieceType;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bag.is_empty() {
            self.refill_bag();
        }

        self.bag.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_generator() {
        let mut generator = PieceGenerator::new();
        for _ in 0..14 {
            let piece = generator.next().unwrap();
            println!("{:?}", piece);
        }
    }
}
