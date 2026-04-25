use crate::engine::pieces::PieceType;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use std::iter::Iterator;

pub struct PieceGenerator {
    bag: Vec<PieceType>,
    rng: StdRng,
}

impl PieceGenerator {
    pub fn new() -> Self {
        let seed = rand::rng().random();
        Self::with_seed(seed)
    }

    pub fn with_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bag = Vec::from(PieceType::all());
        bag.shuffle(&mut rng);

        Self { bag, rng }
    }

    fn refill_bag(&mut self) {
        let mut next_bag = Vec::from(PieceType::all());
        next_bag.shuffle(&mut self.rng);
        next_bag.append(&mut self.bag);

        self.bag = next_bag;
    }

    pub(crate) fn preview(&self) -> Vec<PieceType> {
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

        let next_piece = self.bag.pop();
        if self.bag.len() < PieceType::LEN {
            self.refill_bag();
        }
        next_piece
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_yields_each_piece_once_per_bag() {
        let mut generator = PieceGenerator::with_seed(0);
        let mut pieces = (0..PieceType::LEN)
            .map(|_| generator.next().unwrap())
            .collect::<Vec<_>>();
        pieces.sort_by_key(|piece_type| *piece_type as u8);

        assert_eq!(pieces, PieceType::all());
    }

    #[test]
    fn preview_does_not_consume_the_bag() {
        let mut generator = PieceGenerator::with_seed(0);
        let preview = generator.preview();

        assert_eq!(preview.len(), PieceType::LEN);
        assert_eq!(generator.next(), preview.first().copied());
    }

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut left = PieceGenerator::with_seed(42);
        let mut right = PieceGenerator::with_seed(42);

        let left_sequence = (0..1000).map(|_| left.next().unwrap()).collect::<Vec<_>>();
        let right_sequence = (0..1000).map(|_| right.next().unwrap()).collect::<Vec<_>>();

        assert_eq!(left_sequence, right_sequence);
    }

    #[test]
    fn every_dealt_bag_contains_all_piece_types() {
        let mut generator = PieceGenerator::with_seed(42);

        for _ in 0..20 {
            let mut pieces = (0..PieceType::LEN)
                .map(|_| generator.next().unwrap())
                .collect::<Vec<_>>();
            pieces.sort_by_key(|piece_type| *piece_type as u8);

            assert_eq!(pieces, PieceType::all());
        }
    }
}
