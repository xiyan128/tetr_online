/**
 * Local dev server. Run with hot reload via `bun run dev`.
 *
 * Serves the TSX page (`web/index.html`) through Bun's bundler so editing
 * `web/main.tsx` hot-reloads in the browser. The two runtime dependencies that
 * are NOT part of the JS module graph — the prebuilt wasm bundles and the game
 * `assets/` — are served straight from disk by the `fetch` fallback:
 *
 *   - wasm bundles  <- .wasm-dev/   (produced by `bun run build:wasm`)
 *   - assets/*      <- assets/      (the repo's asset dir, served as-is)
 *
 * `bun run dev` builds the wasm once (slow) then starts this with `--hot`, so
 * iterating on the page is a fast TSX-only loop. Rebuild wasm only when the Rust
 * changes (`bun run build:wasm`).
 */
import index from "./web/index.html";
import { RENDERERS, WASM_DEV_DIR, bundleJs } from "./web/bundles";

const GLUE_FILES = new Set(RENDERERS.map((r) => `/${bundleJs(r)}`));

const server = Bun.serve({
  port: Number(process.env.PORT ?? 8080),
  development: true,
  routes: {
    "/": index,
  },
  async fetch(req) {
    const path = new URL(req.url).pathname;

    // Game assets straight from the repo (matches Bevy's `assets/...` fetches).
    if (path.startsWith("/assets/")) {
      return new Response(Bun.file(`.${path}`));
    }
    // Prebuilt wasm glue + binaries from the dev staging dir.
    if (path.endsWith(".wasm") || GLUE_FILES.has(path)) {
      return new Response(Bun.file(`${WASM_DEV_DIR}${path}`));
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(`tetr_online dev server: ${server.url}`);
console.log(`  WebGL2:  ${server.url}?renderer=webgl2`);
console.log(`  WebGPU:  ${server.url}?renderer=webgpu`);
