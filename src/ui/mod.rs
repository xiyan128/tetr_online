//! Shared UI toolkit for menu screens.
//!
//! * [`widgets`] — themed bundle builders (root, title/label, focusable button).
//! * [`focus`] — keyboard focus-navigation helper (Up/Down move, Enter select,
//!   Esc back) reused by every screen and the menu feature agents.
//!
//! [`theme`] is re-exported here for convenience.

pub mod focus;
pub mod widgets;

pub use widgets::theme;
