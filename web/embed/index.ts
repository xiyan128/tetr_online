/**
 * Public API for the embeddable AI-autoplay Tetris board.
 *
 * Three ways to use it:
 *   - Preact/React:   `import { TetrisGame } from "tetris-embed"`
 *   - imperative:     `import { mount } from "tetris-embed"; mount(el, { theme: "blog" })`
 *   - HTML, no JS:    `<script type="module" src="tetris-embed.js"></script>` then
 *                     `<tetris-game theme="fourlol"></tetris-game>` (auto-registered).
 */

export { TetrisGame } from "./component";
export type { TetrisGameProps } from "./component";
export { mount, registerElement } from "./element";
export { THEMES, DEFAULT_THEME, resolveTheme } from "./themes";
export type { Theme, PieceColors } from "./themes";

// Auto-register <tetris-game> so a bare module script makes the element available.
import { registerElement } from "./element";
registerElement();
