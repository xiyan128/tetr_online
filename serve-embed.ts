/**
 * Dev server for the embedded-board demo. Run via `bun run embedded-demo`.
 *
 * Serves the faux-blog demo page through Bun's bundler (so editing `demo.tsx` or any
 * `web/embed/*` module hot-reloads), plus the two runtime files that aren't part of
 * the JS module graph — the wasm-bindgen glue and the `.wasm` binary — straight from
 * the staging dir `bun run build:embed` produced.
 */
import demo from "./web/embed-demo/index.html";
import { EMBED_DEV_DIR, EMBED_NAME } from "./web/bundles";

const GLUE = `/${EMBED_NAME}.js`;

const server = Bun.serve({
  port: Number(process.env.PORT ?? 8081),
  development: true,
  routes: {
    "/": demo,
  },
  async fetch(req) {
    const path = new URL(req.url).pathname;
    // The wasm-bindgen glue + binary from the staging dir (.embed-dev/).
    if (path === GLUE || path.endsWith(".wasm")) {
      return new Response(Bun.file(`${EMBED_DEV_DIR}${path}`));
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(`tetr-embed demo: ${server.url}`);
