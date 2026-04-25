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
        let mut bag = Vec::from(PieceType::all());
        let mut rng = rand::rng();
        bag.shuffle(&mut rng);

        Self { bag }
    }

    fn refill_bag(&mut self) {
        let mut next_bag = Vec::from(PieceType::all());
        let mut rng = rand::rng();
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
    fn generator_yields_each_piece_once_per_bag() {
        let mut generator = PieceGenerator::new();
        let mut pieces = (0..PieceType::LEN)
            .map(|_| generator.next().unwrap())
            .collect::<Vec<_>>();
        pieces.sort_by_key(|piece_type| *piece_type as u8);

        assert_eq!(pieces, PieceType::all());
    }

    #[test]
    fn preview_does_not_consume_the_bag() {
        let mut generator = PieceGenerator::new();
        let preview = generator.preview();

        assert_eq!(preview.len(), PieceType::LEN);
        assert_eq!(generator.next(), preview.first().copied());
    }
}
