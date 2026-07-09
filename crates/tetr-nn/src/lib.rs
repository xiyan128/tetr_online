//! The learned-evaluation subsystem.
//!
//! Four small modules with one job each:
//!
//! - [`obs`] — what the net sees: the two-board observation, its layout
//!   constants, packing, and the crate's one hash. Single source of truth.
//! - [`net`] — the two-board P+V net: weight loading (PyTorch-exported
//!   safetensors + config) and the batched forward. There is one forward; a
//!   batch of one is the scalar case.
//! - [`shards`] — decision records for training: **store what you serve**.
//!   A shard row holds the encoded bytes the net actually consumed, so a
//!   trainer can never disagree with serving about what an input was.
//! - [`serve`] — the bridge into the search: an [`Evaluator`] that encodes
//!   leaves under a per-decision frozen opponent context and scores sibling
//!   groups in one batched forward.
//!
//! [`Evaluator`]: tetr_core::ai::eval::Evaluator

pub mod net;
pub mod obs;
#[cfg(feature = "coreml")]
pub mod ort_backend;
pub mod serve;
pub mod shards;

#[cfg(any(target_os = "macos", feature = "openblas"))]
extern crate blas_src as _;
