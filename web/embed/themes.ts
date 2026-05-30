/**
 * Themes for the embedded board.
 *
 * The visual language is reverse-engineered from four.lol's opener diagrams: flat,
 * slightly muted guideline minos with a thin lighter strip along the top edge (a
 * subtle bevel, not a gradient), drawn crisp with no per-cell gridlines. The
 * `fourlol` preset is the sampled palette; the others retune the same renderer.
 *
 * Piece colours are indexed to match the wasm `piece_index` mapping — `I,O,T,S,Z,J,L`.
 */

export type PieceColors = readonly [
  I: string,
  O: string,
  T: string,
  S: string,
  Z: string,
  J: string,
  L: string,
];

export interface Theme {
  /** 7 base mino colours, indexed I,O,T,S,Z,J,L (matches the wasm `piece_index`). */
  readonly pieces: PieceColors;
  /**
   * Optional explicit top-edge bevel colour per piece (same I,O,T,S,Z,J,L order). A
   * non-null entry overrides the auto-derived highlight for that piece; a `null` or
   * missing entry falls back to the luminance-aware lighten/darken of the base
   * colour. Use it where the derived tint reads wrong — e.g. the yellow O, whose
   * high luminance would otherwise darken instead of brighten.
   */
  readonly highlights?: ReadonlyArray<string | null>;
  /** 0..1: how far the top-edge highlight strip is lightened toward white. */
  readonly highlightStrength: number;
  /** 0..1: opacity of the ghost (hard-drop preview) cells. */
  readonly ghostAlpha: number;
  /** px inset between cells; `0` is flush (four.lol fidelity), `1`–`2` reads cleaner. */
  readonly cellGap: number;
  readonly board: {
    /** Board fill; `"transparent"` lets the page background show through. */
    readonly background: string;
    /** Faint gridline colour, or `null` for none (four.lol draws none). */
    readonly grid: string | null;
    /** Border stroke colour, or `null` for none. */
    readonly frame: string | null;
    /** Board corner radius in px. */
    readonly radius: number;
  };
  /** Text colour for HUD labels (NEXT / HOLD / score). */
  readonly text: string;
  /** Muted text colour for secondary HUD. */
  readonly textDim: string;
}

/** four.lol's exact sampled palette (T re-derived in their muted register). */
const FOURLOL_PIECES: PieceColors = [
  "#42afe1", // I
  "#f6d03c", // O
  "#a24bd0", // T
  "#51b84d", // S
  "#eb4f65", // Z
  "#1165b5", // J
  "#f38927", // L
];

/**
 * Per-piece bevel overrides for the guideline palette. Only the yellow O needs one:
 * its base is light enough that the auto-derived bevel would *darken* it (muddy),
 * so pin a brighter top face. Index order is I,O,T,S,Z,J,L; `null` = auto-derive.
 */
const O_BEVEL = "#fdf752";
const GUIDELINE_HIGHLIGHTS: ReadonlyArray<string | null> = [null, O_BEVEL, null, null, null, null, null];

/** Brighter, fully-saturated guideline colours for light backgrounds. */
const BRIGHT_PIECES: PieceColors = [
  "#31c7e8", // I
  "#f7d33e", // O
  "#b14ad6", // T
  "#4cd964", // S
  "#ff4d5e", // Z
  "#2f7bf0", // J
  "#ff9f43", // L
];

export const THEMES: Record<string, Theme> = {
  /** Default: four.lol dark slate, flat muted minos, no gridlines. */
  fourlol: {
    pieces: FOURLOL_PIECES,
    highlights: GUIDELINE_HIGHLIGHTS,
    highlightStrength: 0.3,
    ghostAlpha: 0.16,
    cellGap: 0,
    board: { background: "#23262e", grid: null, frame: "rgba(255,255,255,0.06)", radius: 10 },
    text: "#e7e9ee",
    textDim: "#8b9099",
  },

  /** Light editorial: pale board, soft grid, a hair of cell separation. */
  light: {
    pieces: BRIGHT_PIECES,
    highlights: GUIDELINE_HIGHLIGHTS,
    highlightStrength: 0.34,
    ghostAlpha: 0.2,
    cellGap: 0,
    board: { background: "#f5f6f8", grid: "rgba(20,24,31,0.05)", frame: "rgba(20,24,31,0.1)", radius: 10 },
    text: "#2b3038",
    textDim: "#8a9099",
  },

  /** Tuned to the blog's neutral-gray + blue system (works in light contexts). */
  blog: {
    pieces: FOURLOL_PIECES,
    highlights: GUIDELINE_HIGHLIGHTS,
    highlightStrength: 0.26,
    ghostAlpha: 0.18,
    cellGap: 0,
    board: { background: "#fbfbfc", grid: "rgba(20,24,31,0.04)", frame: "rgba(20,24,31,0.08)", radius: 8 },
    text: "#3a3f47",
    textDim: "#9aa0a8",
  },

  /** Minimalist monochrome: graded grays, one accent-free board. */
  mono: {
    pieces: ["#cfd4da", "#aeb4bc", "#c4c9cf", "#9aa0a8", "#878d95", "#b6bbc2", "#9ea4ac"],
    highlightStrength: 0.22,
    ghostAlpha: 0.14,
    cellGap: 0,
    board: { background: "#1c1e23", grid: null, frame: "rgba(255,255,255,0.05)", radius: 10 },
    text: "#d7dade",
    textDim: "#7d828b",
  },
};

export const DEFAULT_THEME = "fourlol";

/** Resolve a theme name, a partial override object, or `undefined` to a full Theme. */
export function resolveTheme(theme?: string | Partial<Theme>): Theme {
  if (!theme) return THEMES[DEFAULT_THEME];
  if (typeof theme === "string") return THEMES[theme] ?? THEMES[DEFAULT_THEME];
  return { ...THEMES[DEFAULT_THEME], ...theme, board: { ...THEMES[DEFAULT_THEME].board, ...theme.board } };
}

/**
 * A monochrome theme: every piece drawn in a single `ink` colour on a transparent
 * board. Used while the AI plays so the board reads as a quiet, theme-matching
 * animation — pass the page's own text colour (`getComputedStyle(el).color`) as
 * `ink` and it adapts to light/dark automatically. `base` provides the non-piece
 * fields (ghost alpha, radius, HUD text); defaults to the four.lol theme.
 */
export function monoTheme(ink: string, base: Theme = THEMES[DEFAULT_THEME]): Theme {
  const pieces = Array(7).fill(ink) as unknown as PieceColors;
  return {
    ...base,
    pieces,
    // Clear any per-piece bevel overrides from the base theme (e.g. the yellow O) —
    // every piece is the same ink here, so the renderer must derive one uniform
    // ink-based bevel for all, not paint a stray yellow band on the grey board.
    highlights: undefined,
    // The renderer derives the top-edge bevel from the ink with a contrast-aware
    // shift (lighten dark ink, darken light ink), so the "flat-topped, lit from
    // above" band reads in both light and dark mode. A touch softer than the colour
    // theme so the monochrome widget stays quiet.
    highlightStrength: 0.34,
    // A whisper of a ghost on the transparent footer — present for legibility of the
    // drop target, but never competing with the live pieces.
    ghostAlpha: 0.07,
    board: { ...base.board, background: "transparent", grid: null, frame: null },
    text: ink,
    textDim: ink,
  };
}
