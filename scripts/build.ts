#!/usr/bin/env bun
/**
 * Web build pipeline (replaces shell scripts; run via package.json tasks).
 *
 *   bun scripts/build.ts wasm [--no-opt]   build the two wasm bundles into .wasm-dev/
 *   bun scripts/build.ts web               full production build into dist/
 *
 * Bevy bakes the renderer in at compile time, so each bundle is the same binary
 * built with a different cargo feature: default (`webgl2`) vs `--features webgpu`.
 * Per bundle: cargo build (profile `web`) -> wasm-bindgen (--target web) -> wasm-opt.
 *
 * Requirements: bun, rustup wasm32-unknown-unknown target, wasm-bindgen-cli
 * (version pinned to Cargo.lock), and wasm-opt (binaryen) for optimized builds.
 */
import { $ } from "bun";
import { CRATE, RENDERERS, WASM_DEV_DIR, bundleName, type Renderer } from "../web/bundles";

const WASM_IN = `target/wasm32-unknown-unknown/web/${CRATE}.wasm`;

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

// Cargo features per renderer. Default features => webgl2 (the universal-
// compatibility bundle); webgpu drops that default toggle and enables webgpu.
const CARGO_ARGS: Record<Renderer, string[]> = {
  webgl2: [],
  webgpu: ["--no-default-features", "--features", "webgpu"],
};

async function buildWasm(outDir: string, optimize: boolean): Promise<void> {
  await $`mkdir -p ${outDir}`;
  for (const renderer of RENDERERS) {
    const name = bundleName(renderer);
    const cargoArgs = CARGO_ARGS[renderer];
    console.log(`==> [${name}] cargo build`);
    await $`cargo build --profile web --target wasm32-unknown-unknown ${cargoArgs}`;

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

const cmd = process.argv[2];
switch (cmd) {
  case "wasm":
    await buildWasm(WASM_DEV_DIR, /* optimize */ !process.argv.includes("--no-opt"));
    break;
  case "web":
    await buildWeb();
    break;
  default:
    console.error("usage: bun scripts/build.ts <wasm [--no-opt] | web>");
    process.exit(1);
}
