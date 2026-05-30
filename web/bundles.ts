/**
 * Single source of truth for the two renderer bundles, shared by the build
 * (scripts/build.ts), the dev server (serve.ts), and the page (web/main.tsx).
 * Keeping the names here means a rename or a third renderer is a one-line change
 * instead of three silently-coupled edits.
 */
export const CRATE = "tetr_online";

export const RENDERERS = ["webgl2", "webgpu"] as const;
export type Renderer = (typeof RENDERERS)[number];

/** Dev staging dir where `bun run build:wasm` writes bundles for serve.ts. */
export const WASM_DEV_DIR = ".wasm-dev";

/** wasm-bindgen `--out-name` for a renderer (its `.js` glue + `_bg.wasm` derive from this). */
export const bundleName = (r: Renderer): string => `${CRATE}_${r}`;

/** The JS glue filename the page dynamically imports at runtime. */
export const bundleJs = (r: Renderer): string => `${bundleName(r)}.js`;

// ---- Embed component (the headless engine+AI wasm + Preact/Canvas board) ----

/** The embed crate's wasm-bindgen `--out-name` (`tetr_embed.js` / `_bg.wasm`). */
export const EMBED_NAME = "tetr_embed";

/** Dev staging dir for the embed wasm, served by `serve-embed.ts`. */
export const EMBED_DEV_DIR = ".embed-dev";

/** Distributable output dir: `tetris-embed.js` + the embed wasm. */
export const EMBED_DIST_DIR = "dist-embed";
