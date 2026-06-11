# Tetr Online — a guideline Tetris in Rust + Bevy

<img width="1289" alt="Screenshot" src="https://user-images.githubusercontent.com/17198282/229279595-87174f68-c88e-41cd-81c2-47be0f383f72.png">

A guideline-compliant Tetris built in Rust on the [Bevy](https://bevyengine.org/)
engine. It began as a single-player clone and is growing toward a modern
multiplayer game in the spirit of Jstris and TETR.IO, built on a pure,
deterministic rule engine so that bots, replays, and lockstep netplay can all
share one source of truth.

Try the web demo: https://www.xiyan.dev/tetr_online/

> **Status:** single-player is fully playable — Marathon / Sprint / Ultra, menus,
> options, and high scores — and a built-in AI can play any mode. Local and online
> multiplayer are the next milestones (see [Roadmap](#roadmap)).

## Features

**Gameplay (guideline-compliant):**

- SRS rotation with the full five-test wall kicks, including the §7.5 T-spin
  point-5 override.
- A seeded 7-bag randomizer (reproducible piece order).
- Hold, ghost piece, hard/soft drop, and a configurable lock-down rule
  (Extended / Infinite / Classic).
- Guideline scoring with T-spin and mini-T-spin recognition, Back-to-Back, and
  combos, plus on-screen line-clear / Tetris / T-spin notifications.
- A 10×20 visible field over a 20-row buffer, with block-out and lock-out
  detection.

**Modes and shell:**

- Three variants: **Marathon** (climb to the final level), **Sprint** (40 lines,
  fastest time), and **Ultra** (highest score in two minutes).
- **Versus** — two boards, live garbage exchange under the guideline rules
  (cancellation, capped rising, batch hole streams), with a pending-garbage
  meter per board, gray garbage rows, a 3-2-1-GO countdown, and rematch flow.
  Play against any model from the AI catalog, or seat two bots and watch them
  duel; the seat interface is built for local/remote human-vs-human later
  (see `docs/adr-versus-mode-ui.md`).
- Title and menu flow, pause, persisted per-variant high-score tables, and an
  options screen for remappable keys, next-queue length, hold/ghost toggles,
  lock-down mode, and music/SFX volume.

**AI:**

- A built-in bot plays through the exact same input surface as a human. The
  **Watch AI** menu entry runs it in any variant, with a model picker spanning
  the catalog: a one-piece greedy baseline, deterministic multi-ply beam
  searches (our linear evaluator and a verbatim port of Cold Clear 2's), and
  the strongest model — best-first graph search with transposition over an
  attack-tuned evaluator.
- Searches run as **anytime sessions** sliced cooperatively across frames, so
  even the heaviest model never stalls a frame — the thinking hides inside the
  bot's human-like reaction delay. An adjustable *handicap* (reaction delay
  plus an imperfection rate) keeps every model beatable rather than flawless,
  and the architecture stays model-agnostic: a stronger search — or a neural
  policy — drops in behind the same seams (see
  `docs/adr-ai-compute-architecture.md`).

**Cross-platform:** runs natively on Windows, macOS, and Linux, and in the browser
via WebAssembly with both WebGPU and WebGL2 renderers.

## Architecture

The codebase is split along one hard boundary:

- **`crates/tetr-core`** is the rule core plus the bot — plain Rust with **no
  Bevy anywhere in its graph**. The engine is a pure, deterministic function of
  `(seed, input frames)`: no wall-clock, no thread-local RNG. That purity is
  what makes headless AI evaluation, replays, and future lockstep multiplayer
  possible. The AI plugs in through the same `PlayerController` seam the
  keyboard uses, so keyboard, AI, and a future network/replay source are
  interchangeable.
- **The Bevy host** (`src/`) drives the engine through a small plain-data
  contract — `InputFrame` in, `EngineSnapshot` and `EngineEvent` out — and owns
  rendering, audio, input, menus, persistence, and the Watch-AI model registry.
- **`crates/tetr-embed`** wraps the same core as a tiny headless wasm widget;
  **`crates/tetr-research`** is the benchmark harness (APP suites, bot-vs-bot
  versus, the Cold Clear 2 referee).

The engine boundary is held by a guideline acceptance suite under `tests/`.

## Controls

Defaults (all remappable in **Options**):

| Action | Key |
| --- | --- |
| Move left / right | ← / → or A / D |
| Soft drop | ↓ or S |
| Hard drop | Space |
| Rotate clockwise | ↑ or W |
| Rotate counter-clockwise | Z |
| Hold | Left Shift |
| Pause | Esc |

## Getting started

You'll need [Rust](https://www.rust-lang.org/tools/install).

```sh
git clone https://github.com/xiyan128/tetr_online.git
cd tetr_online
cargo run
```

### Web build (WebAssembly)

The web bundles are built with [Bun](https://bun.sh/). You'll also need the
`wasm32-unknown-unknown` Rust target, `wasm-bindgen-cli` (version matched to
`Cargo.lock`), and `wasm-opt` from [Binaryen](https://github.com/WebAssembly/binaryen).

```sh
rustup target add wasm32-unknown-unknown
bun install
bun run dev      # build wasm + hot-reloading dev server
bun run build    # production build into dist/ (WebGPU + WebGL2 bundles)
```

Bevy bakes the graphics backend in at compile time, so the production build
compiles the binary twice — once per renderer — and serves the WebGPU bundle where
it's supported, falling back to WebGL2 elsewhere.

## Development

```sh
cargo test                       # unit tests + the guideline acceptance suite
cargo run --features dev         # in-game ECS inspector overlay (egui)
cargo bench                      # criterion benchmarks (engine + AI)

# AI play-evaluation harness (dev-only; deterministic, never ships):
cargo run --release --example arena_smoke --features arena
```

The `arena` feature is a harness for measuring *how well* a bot plays —
reproducible, variance-aware numbers used to tune and compare AI implementations.
It is gated off so it never compiles into the shipped game.

## Roadmap

- [x] **Engine** — pure, deterministic, guideline-correct, with a full acceptance suite.
- [x] **Single-player** — Marathon / Sprint / Ultra, menus, options, high scores, pause.
- [x] **AI player** — a model-agnostic bot with a tunable handicap and a sandbox mode.
- [x] **Versus** — human-vs-AI and AI-vs-AI on one machine, with guideline attack and
  garbage (engine-owned rules, garbage-aware search seams). Local human-vs-human
  needs a second keymap on the same seat architecture.
- [ ] **Online multiplayer** — deterministic lockstep over a relay server.
- [ ] **Polish** — original assets, replays, spectating, larger formats.

The end goal is any mix of human and AI players against any other mix, locally or
online — on a single engine that stays deterministic the whole way.

## Acknowledgements

- Sound effects from [Techmino](https://github.com/26F-Studio/Techmino), used as
  placeholders to be replaced with original assets before any public release.

This is an independent, open-source project, not affiliated with or endorsed by The
Tetris Company. It implements Tetris-guideline mechanics for educational and
recreational purposes and uses no copyrighted assets from official Tetris games. If
you're a rights holder with a concern, please open an issue.

## Contributing

Contributions are welcome — open an issue or a pull request. The `tests/` acceptance
suite is the regression net, so please keep it green.

## License

Released under the MIT License.
