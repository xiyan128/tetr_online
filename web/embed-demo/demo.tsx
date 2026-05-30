/**
 * The embedded-board demo: a faux blog article that embeds several autoplaying,
 * themeable, click-to-take-over boards — the thing `bun run embedded-demo` serves.
 *
 * It exercises everything the component promises: multiple instances on one page, a
 * live global theme switch (hero + inline boards follow it), independent fixed-theme
 * boards side by side, a decorative non-interactive board, and the autoplay →
 * take-over → resume loop.
 */

import { render } from "preact";
import { useState } from "preact/hooks";

import { TetrisGame } from "../embed/component";
import { THEMES } from "../embed/themes";

const THEME_NAMES = Object.keys(THEMES);

function Demo() {
  const [theme, setTheme] = useState<string>("fourlol");

  return (
    <div class="page">
      <style>{CSS}</style>
      <article>
        <p class="kicker">Engineering note · embeddable widgets</p>
        <h1>A Tetris bot that lives in the page</h1>
        <p class="lede">
          The board below is the real game engine — the same deterministic Rust rule
          core that powers the full app — compiled to a ~100&nbsp;KB WebAssembly module
          with the renderer living in the page. An AI plays it on a loop. Click it and
          you take over; let go and it carries on without you.
        </p>

        <div class="switch" role="group" aria-label="Theme">
          <span class="switch-label">Theme</span>
          {THEME_NAMES.map((name) => (
            <button
              key={name}
              class={`chip ${theme === name ? "chip-on" : ""}`}
              onClick={() => setTheme(name)}
            >
              {name}
            </button>
          ))}
        </div>

        <figure class="hero">
          <TetrisGame theme={theme} height={520} reaction={240} imperfection={0.1} />
          <figcaption>
            The hero board follows the theme switch above. It autoplays; click to play
            it yourself, press <kbd>Esc</kbd> (or click away) to hand it back.
          </figcaption>
        </figure>

        <h2>Why it’s so small</h2>
        <p>
          The full game renders with Bevy and ships as a 14&nbsp;MB WebAssembly bundle —
          far too heavy to drop into an article, let alone several times over. But the
          rule engine and the AI never depended on the renderer; they sit behind a tiny
          <em> snapshot</em> contract. Splitting that core into its own crate lets it
          compile <strong>without</strong> the game engine, so what reaches the page is
          a couple hundred kilobytes of logic plus a Canvas2D drawing routine.
          <span class="floatwrap">
            <TetrisGame theme={theme} height={300} sidebar={false} reaction={320} imperfection={0.18} />
          </span>
          The board to the side is a second, independent instance running the same wasm
          module — no sidebar, a slightly weaker and slower bot, purely decorative. Every
          board on this page is its own engine; they share one downloaded module and
          pause themselves when scrolled out of view.
        </p>

        <h2>One renderer, many looks</h2>
        <p>
          Because the renderer is ordinary page code, theming is just data — a palette
          and a few board colors handed to the component. Here are three fixed themes
          side by side, each its own running game:
        </p>

        <div class="grid">
          <Card label="fourlol"><TetrisGame theme="fourlol" height={300} sidebar={false} reaction={260} /></Card>
          <Card label="light"><TetrisGame theme="light" height={300} sidebar={false} reaction={260} /></Card>
          <Card label="mono"><TetrisGame theme="mono" height={300} sidebar={false} reaction={260} /></Card>
        </div>

        <p class="foot">
          Controls when you take over: <kbd>←</kbd> <kbd>→</kbd> move ·
          <kbd>↓</kbd> soft drop · <kbd>Space</kbd> hard drop ·
          <kbd>↑</kbd>/<kbd>X</kbd> rotate · <kbd>Z</kbd> rotate other way ·
          <kbd>Shift</kbd>/<kbd>C</kbd> hold · <kbd>Esc</kbd> release.
        </p>
      </article>
    </div>
  );
}

function Card({ label, children }: { label: string; children: preact.ComponentChildren }) {
  return (
    <div class="card">
      {children}
      <span class="card-label">{label}</span>
    </div>
  );
}

const CSS = `
  :root { color-scheme: light; }
  body { margin: 0; background: #fbfbfc; }
  .page {
    font: 16px/1.65 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    color: #20242b;
    -webkit-font-smoothing: antialiased;
  }
  article { max-width: 46rem; margin: 0 auto; padding: 4rem 1.5rem 6rem; }
  .kicker { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; font-weight: 600; color: #8a9ab0; margin: 0 0 0.6rem; }
  h1 { font-size: 2.3rem; line-height: 1.12; letter-spacing: -0.02em; margin: 0 0 1rem; }
  h2 { font-size: 1.35rem; letter-spacing: -0.01em; margin: 2.6rem 0 0.8rem; }
  .lede { font-size: 1.18rem; line-height: 1.6; color: #3c424c; margin: 0 0 1.8rem; }
  p { margin: 0 0 1.1rem; }
  em { color: #2b3f63; font-style: italic; }
  a { color: #2f6fe0; }
  kbd {
    font: 600 0.78em ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #eef0f3; border: 1px solid #dcdfe4; border-bottom-width: 2px;
    border-radius: 5px; padding: 0.05em 0.4em; color: #3a4150; white-space: nowrap;
  }
  .switch { display: flex; align-items: center; gap: 0.4rem; margin: 0 0 1.6rem; flex-wrap: wrap; }
  .switch-label { font-size: 0.8rem; font-weight: 600; color: #8a909a; margin-right: 0.2rem; }
  .chip {
    font: 500 0.85rem ui-sans-serif, system-ui, sans-serif; cursor: pointer;
    padding: 0.34rem 0.8rem; border-radius: 999px; border: 1px solid #d9dce1;
    background: #fff; color: #4a5159; transition: all 0.15s ease;
  }
  .chip:hover { border-color: #b9bfc8; }
  .chip-on { background: #20242b; border-color: #20242b; color: #fff; }
  figure { margin: 0; }
  .hero { display: flex; flex-direction: column; align-items: center; gap: 0.7rem; margin: 1.5rem 0 2.4rem; }
  .hero figcaption { font-size: 0.85rem; color: #8a909a; text-align: center; max-width: 30rem; line-height: 1.5; }
  .floatwrap { float: right; margin: 0.2rem 0 0.8rem 1.4rem; }
  .grid { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; margin: 1.4rem 0 2rem; }
  .card { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; }
  .card-label { font: 500 0.78rem ui-monospace, monospace; color: #8a909a; }
  .foot { font-size: 0.9rem; color: #6a717b; border-top: 1px solid #ececef; padding-top: 1.2rem; margin-top: 2.4rem; }
  @media (max-width: 560px) { .floatwrap { float: none; display: block; margin: 1rem auto; text-align: center; } }
`;

render(<Demo />, document.getElementById("app")!);
