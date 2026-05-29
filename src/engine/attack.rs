use crate::engine::board::Board;

/// Result of injecting garbage rows into a board.
///
/// Filled out by [`apply_attack`] once N3.6 implements garbage. Reserved here
/// so future call sites compile against the eventual signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttackOutcome {
    /// Number of rows actually inserted (may be less than requested if the
    /// board cannot accept more garbage without topping out).
    pub rows_inserted: u32,
    /// Column index left empty (the "hole") in each inserted row, mirroring
    /// the `gap_column` argument for callers that want to confirm placement.
    pub gap_column: u8,
}

/// Inject `lines` rows of garbage into `board`, leaving column `gap_column`
/// empty in every inserted row.
///
/// Stub — see roadmap N3.6 for the implementation. Kept as a free function so
/// callers reach the same surface that the production version will expose; we
/// just want the signature reserved.
pub fn apply_attack(board: &mut Board, lines: u32, gap_column: u8) -> AttackOutcome {
    let _ = (board, lines, gap_column);
    unimplemented!("garbage insertion — N3.6");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_attack_is_stubbed_until_n3_6() {
        let mut board = Board::new(10, 20);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_attack(&mut board, 1, 0);
        }));

        let payload = result.err().expect("stub must panic until N3.6 lands");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&'static str>().copied())
            .unwrap_or("");
        assert!(
            message.contains("N3.6"),
            "panic message should reference roadmap item: {message:?}"
        );
    }
}
