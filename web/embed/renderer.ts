/**
 * Canvas2D renderer for the embedded board.
 *
 * Reverse-engineered from four.lol: each mino is a **flat fill plus a thin lighter
 * strip along its top edge** (a subtle bevel, no gradient), drawn crisp. Unlike
 * four.lol's tiny `image-rendering: pixelated` bitmaps, we render **DPR-aware** at
 * full resolution (integer-aligned rects) so it stays sharp on any display while
 * animating a live board. Empty cells are simply not drawn.
 *
 * The renderer is pure: it reads a [`GameView`] (the wasm `Game`'s getters) and a
 * [`PreparedTheme`], and draws. All timing/effect state is owned by the caller (the
 * loop) and passed in as `fx`.
 */

import type { Theme } from "./themes";

/** The subset of the wasm `Game` the renderer reads. */
export interface GameView {
  board_cells(): Int32Array;
  active_cells(): Int32Array;
  ghost_cells(): Int32Array;
  next_queue(): Uint8Array;
  hold(): number;
  active_lock_fraction(): number;
  score(): number;
  lines(): number;
  level(): number;
  game_over(): boolean;
  board_width(): number;
  visible_height(): number;
}

/** Per-frame effect intensities the loop decays over time. */
export interface Fx {
  /** 0..1 white board flash (line clear). */
  flash: number;
  /** 0..1 dim overlay (game over / paused). */
  dim: number;
}

export const NO_FX: Fx = { flash: 0, dim: 0 };

export interface LayoutOpts {
  /** Pixels per cell (CSS px). */
  cell: number;
  /** Draw the HOLD + NEXT rail to the right of the board. */
  sidebar: boolean;
  /** Outer padding around the board (CSS px). */
  pad: number;
}

export interface Layout {
  cell: number;
  cols: number;
  /** The engine's visible height in rows (the play area coordinate space). */
  rows: number;
  /** Rows actually drawn = `rows` + a small top buffer for spawn headroom. */
  drawRows: number;
  width: number;
  height: number;
  boardX: number;
  boardY: number;
  boardW: number;
  boardH: number;
  sidebar: boolean;
  railX: number;
  railW: number;
}

/** A theme with its derived (precomputed) highlight palette, built once per theme. */
export interface PreparedTheme {
  theme: Theme;
  highlight: string[];
}

export function prepareTheme(theme: Theme): PreparedTheme {
  // The bevel band is a lighter shade of each piece — except when a piece colour is
  // already so light that lightening is invisible (e.g. the near-white monochrome ink
  // in dark mode). There, shift the band *darker* instead, so the top-lit edge always
  // reads. A theme may also pin an explicit bevel per piece (`theme.highlights`),
  // which wins over the auto-derivation — used for the yellow O, whose high luminance
  // would otherwise darken into a muddy top face instead of brightening.
  const highlight = theme.pieces.map((c, i) => {
    const override = theme.highlights?.[i];
    if (override) return override;
    return luminance(c) > 0.7 ? darken(c, theme.highlightStrength) : lighten(c, theme.highlightStrength);
  });
  return { theme, highlight };
}

/** Compute pixel geometry for a board of `cols × rows` at the given cell size. */
/**
 * Extra rows drawn above the visible field. This engine spawns pieces at the top of
 * the visible field (with their upper cells in the hidden buffer just above it), so
 * without headroom a freshly spawned piece renders flush against — or clipped by —
 * the very top edge and reads as "overflowing the top". Showing a couple of buffer
 * rows lets pieces appear fully and slide in cleanly.
 */
const TOP_BUFFER_ROWS = 2;

export function computeLayout(cols: number, rows: number, o: LayoutOpts): Layout {
  const { cell, pad } = o;
  const drawRows = rows + TOP_BUFFER_ROWS;
  const boardW = cols * cell;
  const boardH = drawRows * cell;
  const railW = o.sidebar ? Math.round(cell * 4.6) : 0;
  const railGap = o.sidebar ? Math.round(cell * 0.7) : 0;
  const width = pad + boardW + (o.sidebar ? railGap + railW : 0) + pad;
  const height = pad + boardH + pad;
  return {
    cell,
    cols,
    rows,
    drawRows,
    width,
    height,
    boardX: pad,
    boardY: pad,
    boardW,
    boardH,
    sidebar: o.sidebar,
    railX: pad + boardW + railGap,
    railW,
  };
}

/** Size the canvas bitmap for the device pixel ratio and return a DPR-scaled ctx. */
export function setupCanvas(canvas: HTMLCanvasElement, layout: Layout, dpr: number): CanvasRenderingContext2D {
  canvas.width = Math.round(layout.width * dpr);
  canvas.height = Math.round(layout.height * dpr);
  canvas.style.width = `${layout.width}px`;
  canvas.style.height = `${layout.height}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Draw one frame. */
export function drawFrame(ctx: CanvasRenderingContext2D, view: GameView, pt: PreparedTheme, layout: Layout, fx: Fx): void {
  const t = pt.theme;
  const { cell, rows, boardX, boardY, boardW, boardH } = layout;

  ctx.clearRect(0, 0, layout.width, layout.height);

  // Board plate.
  roundRect(ctx, boardX, boardY, boardW, boardH, t.board.radius);
  if (t.board.background !== "transparent") {
    ctx.fillStyle = t.board.background;
    ctx.fill();
  }

  // Clip to the board for the playfield contents.
  ctx.save();
  ctx.clip();

  if (t.board.grid) drawGrid(ctx, layout, t.board.grid);

  // Engine y increases upward with the floor at y = 0, so row 0 sits at the canvas
  // bottom. The canvas is `drawRows` tall (the visible field + a top buffer), so a
  // piece spawning at the top of the field appears with headroom and slides in.
  // (Cells deeper in the hidden buffer map to negative y and are clipped below.)
  const sx = (x: number) => boardX + x * cell;
  const sy = (y: number) => boardY + (layout.drawRows - 1 - y) * cell;

  // Ghost (hard-drop preview): faint flat fill, no bevel.
  const ghost = view.ghost_cells();
  ctx.globalAlpha = t.ghostAlpha;
  for (let i = 0; i < ghost.length; i += 3) {
    drawCell(ctx, sx(ghost[i]), sy(ghost[i + 1]), cell, t.pieces[ghost[i + 2]]);
  }
  ctx.globalAlpha = 1;

  // Locked stack: flat fill plus a lighter band along each shape's *exposed* top
  // faces — a cell with nothing directly above it. Vertically-stacked same-colour
  // cells share no band, so a piece reads as one chunky, top-lit solid with no inner
  // seams (the band is pixel-snapped to the cell, so adjacent bands stay flush).
  const board = view.board_cells();
  const boardOcc = occupancy(board);
  for (let i = 0; i < board.length; i += 3) {
    const x = board[i];
    const y = board[i + 1];
    const idx = board[i + 2];
    drawCell(ctx, sx(x), sy(y), cell, t.pieces[idx]);
    if (t.highlightStrength > 0 && !boardOcc.has(cellKey(x, y + 1))) {
      drawBevel(ctx, sx(x), sy(y), cell, pt.highlight[idx]);
    }
  }

  // Active piece: same treatment, brightening as the lock timer runs out.
  const active = view.active_cells();
  const activeOcc = occupancy(active);
  const lock = view.active_lock_fraction();
  for (let i = 0; i < active.length; i += 3) {
    const x = active[i];
    const y = active[i + 1];
    const idx = active[i + 2];
    const base = lock > 0 ? lighten(t.pieces[idx], lock * 0.35) : t.pieces[idx];
    drawCell(ctx, sx(x), sy(y), cell, base);
    if (t.highlightStrength > 0 && !activeOcc.has(cellKey(x, y + 1))) {
      drawBevel(ctx, sx(x), sy(y), cell, pt.highlight[idx]);
    }
  }

  // Line-clear flash / game-over dim are full-board overlays — meaningful only on an
  // opaque board. On a transparent board they would flash a box over the page, so
  // they are skipped there (the widget stays quiet).
  if (t.board.background !== "transparent") {
    if (fx.flash > 0.001) {
      ctx.fillStyle = `rgba(255,255,255,${(fx.flash * 0.18).toFixed(3)})`;
      ctx.fillRect(boardX, boardY, boardW, boardH);
    }
    if (fx.dim > 0.001) {
      ctx.fillStyle = `rgba(10,12,16,${(fx.dim * 0.55).toFixed(3)})`;
      ctx.fillRect(boardX, boardY, boardW, boardH);
    }
  }

  ctx.restore();

  // Board frame on top of contents.
  if (t.board.frame) {
    roundRect(ctx, boardX + 0.5, boardY + 0.5, boardW - 1, boardH - 1, t.board.radius);
    ctx.strokeStyle = t.board.frame;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  if (layout.sidebar) drawSidebar(ctx, view, pt, layout);
}

// ---- pieces ----

/**
 * A flat, pixel-snapped cell. Rounding each edge to whole pixels makes adjacent
 * cells share an exact boundary, so same-colour tiles tile seamlessly — no
 * anti-aliased hairline and no per-cell bevel between them, so the whole piece reads
 * as one continuous shape.
 */
function drawCell(ctx: CanvasRenderingContext2D, px: number, py: number, cell: number, color: string): void {
  const x0 = Math.round(px);
  const y0 = Math.round(py);
  const x1 = Math.round(px + cell);
  const y1 = Math.round(py + cell);
  ctx.fillStyle = color;
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
}

/**
 * A lighter band along a cell's top edge: the "top-lit, flat-topped" bevel. Drawn
 * only on cells whose top face is exposed (no same-piece cell directly above), so a
 * stacked column shows one band at its crown, not a stripe per cell. Pixel-snapped to
 * the same grid as [`drawCell`] so the band spans the full cell width with no gap.
 */
function drawBevel(ctx: CanvasRenderingContext2D, px: number, py: number, cell: number, color: string): void {
  const x0 = Math.round(px);
  const y0 = Math.round(py);
  const x1 = Math.round(px + cell);
  const band = Math.max(2, Math.round(cell * 0.18));
  ctx.fillStyle = color;
  ctx.fillRect(x0, y0, x1 - x0, band);
}

/** Stable integer key for a board cell `(x, y)` (board is at most 10×40). */
function cellKey(x: number, y: number): number {
  return x * 100 + y;
}

/** Set of occupied `(x, y)` from a packed `[x, y, idx, …]` array, for adjacency tests. */
function occupancy(packed: Int32Array): Set<number> {
  const set = new Set<number>();
  for (let i = 0; i < packed.length; i += 3) set.add(cellKey(packed[i], packed[i + 1]));
  return set;
}

function drawGrid(ctx: CanvasRenderingContext2D, layout: Layout, color: string): void {
  const { cell, cols, rows, boardX, boardY, boardW, boardH } = layout;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 1; c < cols; c++) {
    const x = Math.round(boardX + c * cell) + 0.5;
    ctx.moveTo(x, boardY);
    ctx.lineTo(x, boardY + boardH);
  }
  for (let r = 1; r < rows; r++) {
    const y = Math.round(boardY + r * cell) + 0.5;
    ctx.moveTo(boardX, y);
    ctx.lineTo(boardX + boardW, y);
  }
  ctx.stroke();
}

// Preview shapes (col, row-from-top) in a 4×2 box, indexed I,O,T,S,Z,J,L.
const PREVIEW_SHAPES: ReadonlyArray<ReadonlyArray<readonly [number, number]>> = [
  [[0, 0], [1, 0], [2, 0], [3, 0]], // I
  [[1, 0], [2, 0], [1, 1], [2, 1]], // O
  [[1, 0], [0, 1], [1, 1], [2, 1]], // T
  [[1, 0], [2, 0], [0, 1], [1, 1]], // S
  [[0, 0], [1, 0], [1, 1], [2, 1]], // Z
  [[0, 0], [0, 1], [1, 1], [2, 1]], // J
  [[2, 0], [0, 1], [1, 1], [2, 1]], // L
];

/** Draw a centered mini-piece preview within a box. */
function drawPreview(ctx: CanvasRenderingContext2D, idx: number, bx: number, by: number, bw: number, bh: number, pt: PreparedTheme): void {
  if (idx < 0 || idx > 6) return;
  const shape = PREVIEW_SHAPES[idx];
  let maxC = 0;
  let maxR = 0;
  for (const [c, r] of shape) {
    if (c > maxC) maxC = c;
    if (r > maxR) maxR = r;
  }
  const pcell = Math.min(bw / (maxC + 1.6), bh / (maxR + 1.6));
  const offX = bx + (bw - (maxC + 1) * pcell) / 2;
  const offY = by + (bh - (maxR + 1) * pcell) / 2;
  for (const [c, r] of shape) {
    drawCell(ctx, offX + c * pcell, offY + r * pcell, pcell, pt.theme.pieces[idx]);
  }
}

function drawSidebar(ctx: CanvasRenderingContext2D, view: GameView, pt: PreparedTheme, layout: Layout): void {
  const t = pt.theme;
  const { railX, railW, boardY, cell } = layout;
  ctx.textBaseline = "alphabetic";
  ctx.font = `600 ${Math.round(cell * 0.5)}px ui-sans-serif, system-ui, sans-serif`;

  let y = boardY;
  const box = railW;
  const previewH = Math.round(cell * 2.4);

  // HOLD
  ctx.fillStyle = t.textDim;
  ctx.fillText("HOLD", railX, y + cell * 0.5);
  y += cell * 0.7;
  plate(ctx, railX, y, box, previewH, t);
  drawPreview(ctx, view.hold(), railX, y, box, previewH, pt);
  y += previewH + cell * 0.7;

  // NEXT (up to 3)
  ctx.fillStyle = t.textDim;
  ctx.fillText("NEXT", railX, y + cell * 0.5);
  y += cell * 0.7;
  const queue = view.next_queue();
  const count = Math.min(3, queue.length);
  for (let i = 0; i < count; i++) {
    plate(ctx, railX, y, box, previewH, t);
    drawPreview(ctx, queue[i], railX, y, box, previewH, pt);
    y += previewH + cell * 0.35;
  }
}

function plate(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, t: Theme): void {
  roundRect(ctx, x, y, w, h, Math.min(8, t.board.radius));
  if (t.board.background !== "transparent") {
    ctx.fillStyle = t.board.background;
    ctx.fill();
  }
  if (t.board.frame) {
    ctx.strokeStyle = t.board.frame;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// ---- utils ----

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rad = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function lighten(color: string, amt: number): string {
  const { r, g, b } = parseColor(color);
  const mix = (c: number) => Math.round(c + (255 - c) * amt);
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

function darken(color: string, amt: number): string {
  const { r, g, b } = parseColor(color);
  const mix = (c: number) => Math.round(c * (1 - amt));
  return `rgb(${mix(r)},${mix(g)},${mix(b)})`;
}

/** Relative luminance in `0..1` (perceptual weights), for choosing bevel direction. */
function luminance(color: string): number {
  const { r, g, b } = parseColor(color);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

/**
 * Parse `#rgb`, `#rrggbb`, or `rgb()/rgba()` into channels. Accepting the `rgb()`
 * form matters because the monochrome AI palette is derived from the page's computed
 * text colour (`getComputedStyle(...).color`), which browsers return as `rgb(...)`.
 */
function parseColor(color: string): { r: number; g: number; b: number } {
  const c = color.trim();
  if (c.startsWith("#")) {
    let h = c.slice(1);
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    const n = parseInt(h, 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }
  const m = c.match(/rgba?\(([^)]+)\)/i);
  if (m) {
    // Accept both the legacy comma form `rgb(r, g, b)` and CSS Color 4's
    // space-separated `rgb(r g b / a)`: split on commas, slashes, and whitespace.
    const [r, g, b] = m[1].split(/[\s,/]+/).filter(Boolean).map((s) => parseFloat(s));
    return { r: r || 0, g: g || 0, b: b || 0 };
  }
  return { r: 136, g: 136, b: 136 };
}
