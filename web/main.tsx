/**
 * The entire web page for tetr_online.
 *
 * Boots the Bevy wasm game, choosing a renderer at runtime. Bevy compiles the
 * graphics backend in at build time, so we ship two bundles — `tetr_online_webgpu`
 * and `tetr_online_webgl2` — and pick one here:
 *
 *   1. `?renderer=webgpu|webgl2` forces a bundle (for testing both paths).
 *   2. Otherwise prefer WebGPU, but only after confirming a real adapter is
 *      reachable (mere `navigator.gpu` presence doesn't guarantee a working
 *      device — it can be blocklisted or fail to create).
 *   3. If the WebGPU bundle throws during init, fall back to WebGL2 so a flaky
 *      WebGPU stack never leaves the player with a blank canvas.
 *
 * WebGL2 is the universal floor: anything that runs this game at all supports it.
 */
import { render } from "preact";
import { useEffect, useState } from "preact/hooks";
import { RENDERERS, bundleJs, type Renderer } from "./bundles";

/** wasm-bindgen `--target web` module shape: default export is the async init(). */
type WasmModule = { default: (module_or_path?: string) => Promise<unknown> };

function forced(): Renderer | null {
  const r = new URLSearchParams(location.search).get("renderer");
  return RENDERERS.includes(r as Renderer) ? (r as Renderer) : null;
}

async function pickRenderer(): Promise<Renderer> {
  const f = forced();
  if (f) return f;
  if (!navigator.gpu) return "webgl2";
  try {
    return (await navigator.gpu.requestAdapter()) ? "webgpu" : "webgl2";
  } catch {
    return "webgl2";
  }
}

async function loadBundle(renderer: Renderer): Promise<void> {
  // Resolve against the document base URL so the wasm bundle (which sits next to
  // index.html, NOT next to the hashed JS chunk) is found under any base path.
  // The URL is computed at runtime, so the bundler leaves this import external.
  const url = new URL(bundleJs(renderer), document.baseURI).href;
  const mod = (await import(url)) as WasmModule;
  await mod.default();
}

/** Pick a renderer, boot it, and fall back WebGPU -> WebGL2 on init failure. */
async function boot(): Promise<Renderer> {
  let renderer = await pickRenderer();
  try {
    await loadBundle(renderer);
  } catch (err) {
    if (renderer === "webgpu" && !forced()) {
      console.warn("WebGPU bundle failed to start, falling back to WebGL2:", err);
      renderer = "webgl2";
      await loadBundle(renderer);
    } else {
      throw err;
    }
  }
  return renderer;
}

function App() {
  const [status, setStatus] = useState<"loading" | "running" | "error">("loading");

  // Boots exactly once: App is mounted at the document root and never unmounts.
  useEffect(() => {
    boot()
      .then((renderer) => {
        console.info(`tetr_online: started with ${renderer} renderer`);
        setStatus("running");
      })
      .catch((err) => {
        console.error("tetr_online failed to start:", err);
        setStatus("error");
      });
  }, []);

  return (
    <div id="wrap">
      {/* Bevy binds to this canvas (`canvas: Some("#bevy")` in main.rs) and
          resizes it to the parent (`fit_canvas_to_parent: true`). It must stay
          mounted for the lifetime of the app so Bevy's handle stays valid. */}
      <canvas id="bevy">Javascript and support for canvas is required</canvas>
      {status !== "running" && (
        <div id="overlay">
          {status === "error" ? "Failed to start. See console for details." : "Loading…"}
        </div>
      )}
    </div>
  );
}

render(<App />, document.getElementById("app")!);
