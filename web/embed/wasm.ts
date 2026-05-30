/**
 * Loads the wasm-bindgen `--target web` glue once and exposes the `Game` class.
 *
 * Mirrors the game page's loader pattern: the glue URL is computed at *runtime* so
 * the bundler leaves the `import()` external, and the glue resolves its `_bg.wasm`
 * relative to itself. Initialization is idempotent — many boards on a page share one
 * wasm module.
 */

import type { GameView } from "./renderer";

/** The full wasm `Game` surface (render getters from [`GameView`] plus controls). */
export interface WasmGame extends GameView {
  tick(dt: number): Uint8Array;
  set_mode_ai(): void;
  set_mode_human(): void;
  is_human(): boolean;
  key_down(action: number): void;
  key_up(action: number): void;
  reset(seed: number): void;
  active_piece(): number;
  back_to_back(): boolean;
  /** Free the wasm-owned memory (call on unmount). */
  free(): void;
}

export interface GameCtor {
  new (seed: number, reactionMs: number, imperfection: number): WasmGame;
}

interface GlueModule {
  default: (input?: unknown) => Promise<unknown>;
  Game: GameCtor;
}

let modPromise: Promise<GlueModule> | null = null;

/**
 * Import + initialize the wasm glue and return the `Game` constructor.
 *
 * `glueUrl` defaults to `tetr_embed.js` resolved against the document base; pass an
 * explicit URL to embed under a different base path. Safe to call repeatedly.
 */
export async function loadGameClass(glueUrl?: string): Promise<GameCtor> {
  if (!modPromise) {
    modPromise = (async () => {
      const base = typeof document !== "undefined" ? document.baseURI : location.href;
      const url = new URL(glueUrl ?? "tetr_embed.js", base).href;
      // Runtime-computed specifier keeps this import external to the bundle.
      const mod = (await import(/* @vite-ignore */ /* webpackIgnore: true */ url)) as GlueModule;
      await mod.default();
      return mod;
    })();
  }
  return (await modPromise).Game;
}
