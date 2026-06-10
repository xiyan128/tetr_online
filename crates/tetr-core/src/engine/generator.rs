//! Seven-bag piece generator.
//!
//! Yields each of the seven tetrominoes once per "bag" before reshuffling, the
//! guideline-standard randomizer. Implemented as an [`Iterator`] so callers can
//! pull pieces lazily; the bag is refilled transparently and kept topped up so a
//! short preview window is always available.

use crate::engine::pieces::PieceType;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub struct PieceGenerator {
    /// Deal stack: pieces pop from the END. Layout invariant (kept by `with_seed`
    /// pre-filling one bag ahead and `next` refilling whenever `len` drops below
    /// 7): the first 7 elements are the *next* (untouched) bag, and everything at
    /// index 7.. is the not-yet-dealt remainder of the *current* bag — so
    /// `bag[7..]` IS [`bag_remainder`](Self::bag_remainder), and an exactly-empty
    /// remainder (`len == 7`) is a bag boundary.
    bag: Vec<PieceType>,
    rng: StdRng,
}

impl PieceGenerator {
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bag = Vec::from(PieceType::all());
        bag.shuffle(&mut rng);

        let mut generator = Self { bag, rng };
        // Pre-fill one bag ahead so the layout invariant above holds from the
        // start (the fresh, untouched current bag sits at index 7..). This draws
        // the second bag's shuffle at construction instead of on the first
        // `next()`; the shuffles are consumed in the same order either way, so
        // the dealt piece sequence for a given seed is unchanged.
        generator.refill_bag();
        generator
    }

    fn refill_bag(&mut self) {
        let mut next_bag = Vec::from(PieceType::all());
        next_bag.shuffle(&mut self.rng);
        next_bag.append(&mut self.bag);

        self.bag = next_bag;
    }

    /// The not-yet-dealt remainder of the **current** 7-bag, i.e. exactly the set
    /// the next [`next()`](Iterator::next) call deals from — empty at a bag
    /// boundary (the next deal opens a fresh bag of all seven). This is the
    /// engine-side ground truth a search needs to speculate past the revealed
    /// queue; reconstructing it from the queue alone is impossible (the queue
    /// window straddles bag boundaries).
    pub fn bag_remainder(&self) -> &[PieceType] {
        &self.bag[7..]
    }

    /// The next `PieceType::LEN` pieces, in deal order, without consuming the
    /// bag. Used to assert preview invariants; production preview is served by
    /// the engine's own look-ahead queue, so this is test-only.
    #[cfg(test)]
    fn preview(&self) -> Vec<PieceType> {
        self.bag[self.bag.len() - PieceType::LEN..]
            .iter()
            .rev()
            .copied()
            .collect()
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

    /// Sorted copy of a piece set, for order-insensitive comparison.
    fn sorted(pieces: &[PieceType]) -> Vec<PieceType> {
        let mut pieces = pieces.to_vec();
        pieces.sort_by_key(|piece_type| *piece_type as u8);
        pieces
    }

    #[test]
    fn bag_remainder_is_the_complement_of_the_current_bags_dealt_pieces() {
        // The exported remainder must equal "all seven minus what the current bag
        // has dealt" at every point, across several bag boundaries. The dealt
        // stream itself is the ground truth: piece `i` belongs to bag `i / 7`.
        let mut generator = PieceGenerator::with_seed(7);

        // Fresh generator: nothing dealt, the whole current bag remains.
        assert_eq!(sorted(generator.bag_remainder()), PieceType::all());

        let mut dealt: Vec<PieceType> = Vec::new();
        for i in 0usize..21 {
            dealt.push(generator.next().unwrap());
            // At an exact bag boundary the current bag is spent and the export is
            // EMPTY (the next deal opens a fresh bag); otherwise it is the
            // complement of what the current bag has dealt so far.
            let expected: Vec<PieceType> = if (i + 1).is_multiple_of(7) {
                Vec::new()
            } else {
                let bag_start = ((i + 1) / 7) * 7;
                let dealt_this_bag = &dealt[bag_start..];
                PieceType::all()
                    .into_iter()
                    .filter(|pt| !dealt_this_bag.contains(pt))
                    .collect()
            };
            assert_eq!(
                sorted(generator.bag_remainder()),
                sorted(&expected),
                "remainder mismatch after deal {i}"
            );
        }
    }

    #[test]
    fn bag_remainder_is_empty_exactly_at_bag_boundaries() {
        let mut generator = PieceGenerator::with_seed(0);
        for i in 1usize..=21 {
            generator.next().unwrap();
            assert_eq!(
                generator.bag_remainder().is_empty(),
                i.is_multiple_of(7),
                "boundary emptiness wrong after {i} deals"
            );
        }
    }
}
