#!/usr/bin/env bun
/**
 * Web build pipeline (replaces shell scripts; run via package.json tasks).
 *
 *   bun scripts/build.ts wasm [--no-opt]   build the two wasm bundles into .wasm-dev/
 *   bun scripts/build.ts web               full production build into dist/
 *
 * Bevy bakes the renderer in at compile time, so each bundle is the same binary
 * built with a different cargo feature: `webgl2` vs `webgpu` (both with
 * --no-default-features). Neither bundle ships the optional `bloom` skin —
 * Kissaten is flat by rule; bloom is opt-in via `--features bloom` only.
 * Per bundle: cargo build (profile `web`) -> wasm-bindgen (--target web) -> wasm-opt.
 *
 * Requirements: bun, rustup wasm32-unknown-unknown target, wasm-bindgen-cli
 * (version pinned to Cargo.lock), and wasm-opt (binaryen) for optimized builds.
 */
import { $ } from "bun";
import {
  CRATE,
  EMBED_DEV_DIR,
  EMBED_DIST_DIR,
  EMBED_NAME,
  RENDERERS,
  WASM_DEV_DIR,
  bundleName,
  type Renderer,
} from "../web/bundles";

const WASM_IN = `target/wasm32-unknown-unknown/web/${CRATE}.wasm`;

/**
 * Fail loudly if the installed `wasm-bindgen` CLI doesn't match the version the build
 * links (pinned in `Cargo.lock`). A mismatch silently produces glue that's subtly
 * incompatible with the generated `.wasm` (cryptic runtime errors, not a build
 * failure), which is exactly the kind of "works on my machine" trap a reproducible
 * build must rule out. Runs once per build invocation; cached after the first call.
 */
let checkedBindgen = false;
async function assertWasmBindgenVersion(): Promise<void> {
  if (checkedBindgen) return;
  checkedBindgen = true;

  // The version Cargo.lock pins for the `wasm-bindgen` crate (the glue must match).
  const lock = await Bun.file("Cargo.lock").text();
  const locked = lock.match(/name = "wasm-bindgen"\nversion = "([^"]+)"/)?.[1];
  if (!locked) {
    console.warn("==> [warn] could not read wasm-bindgen version from Cargo.lock; skipping check");
    return;
  }

  // The installed CLI version (`wasm-bindgen 0.2.118` → `0.2.118`).
  const cli = (await $`wasm-bindgen --version`.text()).trim().split(/\s+/).at(-1);
  if (cli !== locked) {
    throw new Error(
      `wasm-bindgen CLI ${cli} != Cargo.lock ${locked}. The CLI and the linked crate ` +
        `must match or the generated glue is incompatible with the .wasm. ` +
        `Install the matching CLI: cargo install -f wasm-bindgen-cli --version ${locked}`,
    );
  }
}

// Recent rustc emits these wasm features by default, so wasm-opt must be told to
// accept them or validation fails. They only *permit* features the binary already
// uses (wasm-opt never introduces new ones); all are baseline browser support
// since ~2021, so there's no compatibility regression.
const WASM_OPT_FLAGS = [
  "--enable-bulk-memory",
  "--enable-nontrapping-float-to-int",
  "--enable-sign-ext",
  "--enable-mutable-globals",
  "--enable-reference-types",
  "--enable-multivalue",
];

// Cargo features per renderer. Both bundles build with `--no-default-features`
// and name their renderer explicitly because Bevy bakes the wasm backend in at
// compile time. Bloom is part of neither: it is an opt-in skin feature
// (`--features bloom`), and the Kissaten core look is flat by rule.
const CARGO_ARGS: Record<Renderer, string[]> = {
  webgl2: ["--no-default-features", "--features", "webgl2"],
  webgpu: ["--no-default-features", "--features", "webgpu"],
};

async function buildWasm(outDir: string, optimize: boolean): Promise<void> {
  await $`mkdir -p ${outDir}`;
  for (const renderer of RENDERERS) {
    const name = bundleName(renderer);
    const cargoArgs = CARGO_ARGS[renderer];
    console.log(`==> [${name}] cargo build`);
    await $`cargo build --locked --profile web --target wasm32-unknown-unknown ${cargoArgs}`;

    console.log(`==> [${name}] wasm-bindgen`);
    await $`wasm-bindgen --no-typescript --target web --out-dir ${outDir} --out-name ${name} ${WASM_IN}`;

    const wasm = `${outDir}/${name}_bg.wasm`;
    if (optimize) {
      console.log(`==> [${name}] wasm-opt -Oz`);
      await $`wasm-opt -Oz ${WASM_OPT_FLAGS} --output ${wasm} ${wasm}`;
    }
  }
  console.log(`==> wasm bundles ready in ${outDir}/`);
}

async function buildWeb(): Promise<void> {
  const out = "dist";
  console.log(`==> Cleaning ${out}/`);
  await $`rm -rf ${out}`;

  // 1. wasm bundles straight into dist/.
  await buildWasm(out, /* optimize */ true);

  // 2. Bundle + minify the TSX page. Bun emits dist/index.html referencing a
  //    hashed JS chunk; the runtime-computed import() of the wasm glue stays
  //    external, so the wasm bundles above are loaded at runtime.
  console.log("==> bun build (TSX page)");
  const result = await Bun.build({
    entrypoints: ["web/index.html"],
    outdir: out,
    minify: true,
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error("bun build (TSX page) failed");
  }

  // 3. Game assets next to index.html (Bevy fetches `assets/...` relative to the
  //    document base URL).
  console.log("==> Copying assets");
  await $`cp -R assets ${out}/assets`;
  await $`rm -f ${out}/assets/.DS_Store`;

  console.log("==> Done. dist/ contents:");
  await $`ls -la ${out}`;
}

// The embed crate compiles WITHOUT Bevy, so its wasm is a few hundred KB (vs the
// game's ~14 MB) and builds in seconds. Same pipeline: cargo -> wasm-bindgen -> wasm-opt.
const EMBED_WASM_IN = `target/wasm32-unknown-unknown/web/${EMBED_NAME}.wasm`;

async function buildEmbedWasm(outDir: string, optimize: boolean): Promise<void> {
  await $`mkdir -p ${outDir}`;
  console.log(`==> [${EMBED_NAME}] cargo build (no Bevy)`);
  await $`cargo build --locked -p tetr-embed --profile web --target wasm32-unknown-unknown`;

  console.log(`==> [${EMBED_NAME}] wasm-bindgen`);
  await $`wasm-bindgen --no-typescript --target web --out-dir ${outDir} --out-name ${EMBED_NAME} ${EMBED_WASM_IN}`;

  const wasm = `${outDir}/${EMBED_NAME}_bg.wasm`;
  if (optimize) {
    console.log(`==> [${EMBED_NAME}] wasm-opt -Oz`);
    await $`wasm-opt -Oz ${WASM_OPT_FLAGS} --output ${wasm} ${wasm}`;
  }
  console.log(`==> embed wasm ready in ${outDir}/ (${(Bun.file(wasm).size / 1024) | 0} KB)`);
}

// Distributable bundle: the optimized wasm + a single minified `tetris-embed.js`
// (Preact bundled in) you can drop onto any page.
async function buildEmbedDist(): Promise<void> {
  const out = EMBED_DIST_DIR;
  console.log(`==> Cleaning ${out}/`);
  await $`rm -rf ${out}`;
  await buildEmbedWasm(out, /* optimize */ true);

  console.log("==> bun build (tetris-embed.js)");
  const result = await Bun.build({
    entrypoints: ["web/embed/index.ts"],
    outdir: out,
    naming: "tetris-embed.[ext]",
    minify: true,
    format: "esm",
  });
  if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error("bun build (tetris-embed.js) failed");
  }
  console.log(`==> Done. ${out}/ contents:`);
  await $`ls -la ${out}`;
}

const cmd = process.argv[2];
switch (cmd) {
  case "wasm":
    await buildWasm(WASM_DEV_DIR, /* optimize */ !process.argv.includes("--no-opt"));
    break;
  case "web":
    await buildWeb();
    break;
  case "embed":
    await buildEmbedWasm(EMBED_DEV_DIR, /* optimize */ !process.argv.includes("--no-opt"));
    break;
  case "embed-dist":
    await buildEmbedDist();
    break;
  default:
    console.error("usage: bun scripts/build.ts <wasm [--no-opt] | web | embed [--no-opt] | embed-dist>");
    process.exit(1);
}
