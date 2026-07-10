//! The learned-evaluation subsystem.
//!
//! Four small modules with one job each:
//!
//! - [`obs`] — what the net sees: the observation, its layout constants,
//!   packing, and the crate's one hash. Single source of truth.
//! - [`net`] — the value net: weight loading (PyTorch-exported safetensors +
//!   config) and the batched forward. There is one forward; a batch of one is
//!   the scalar case.
//! - [`shards`] — training rows: **store what you serve**. A shard row holds
//!   the encoded bytes the net actually consumed plus the game outcome, so a
//!   trainer can never disagree with serving about what an input was.
//! - [`serve`] — the bridge into the search: an [`Evaluator`] that scores
//!   sibling groups in one batched forward, pure `z_scale · z_hat`.
//!
//! [`Evaluator`]: tetr_core::ai::eval::Evaluator

pub mod net;
pub mod obs;
pub mod serve;
pub mod shards;

#[cfg(any(target_os = "macos", feature = "openblas"))]
extern crate blas_src as _;
